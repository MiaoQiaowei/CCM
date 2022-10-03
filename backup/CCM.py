import torch
import torch.nn as nn
import torch.nn.functional as F
from domainbed import networks


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        )

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class CCM(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.Q_encoder = networks.Featurizer(input_shape, self.hparams)
        self.K_encoder = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.Q_encoder.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        )
        self.head = nn.Sequential(
            nn.Linear(self.Q_encoder.n_outputs, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )
        self.mixer = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.Q_encoder.n_outputs),
        )

        self.num_classes = num_classes
        self.num_domains = num_domains
        self.bs = hparams["batch_size"]
        self.embed_num = self.bs * 4 * num_domains
        self.embed_dim = self.Q_encoder.n_outputs
        self.t = 0.07
        self.m = 0.999

        params = [
            {"params": self.classifier.parameters()},
            {"params": self.Q_encoder.parameters()},
            {"params": self.head.parameters()},
            {"params": self.mixer.parameters()},
        ]

        self.optimizer = torch.optim.Adam(
            params, lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"]
        )

        self.register_buffer("queue", torch.randn(self.embed_dim, self.embed_num))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_label", torch.arange(0, self.embed_num))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.m_up_flag = False

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        all_x_list = all_x.chunk(self.num_domains, dim=0)
        all_y_list = all_y.chunk(self.num_domains, dim=0)

        loss_avg = 0
        loss_learn_avg = 0
        loss_teach_avg = 0
        loss_self_avg = 0

        for x, y, d in zip(all_x_list, all_y_list, range(self.num_domains)):
            x_sd = x
            y_sd = y

            f_sd = self.Q_encoder(x_sd)
            with torch.no_grad():
                self._momentum_update_key_encoder(self.m_up_flag)
                f_sd_k = self.K_encoder(x_sd)

            # learn loss
            CE = self.causal(
                all_f=f_sd, embedding=self.queue.clone().detach().T, all_y=y_sd
            )
            loss_learn = F.cross_entropy(CE, y_sd)

            # teach loss
            logits = self.classifier(f_sd)
            loss_teach = F.cross_entropy(logits, y_sd)

            # self loss
            self_attn = self.attn_score(f_sd, self.queue.clone().detach().T)  # bs e_num
            self_label = (
                y_sd.unsqueeze(dim=1).repeat(1, self.embed_num)
                == self.queue_label.unsqueeze(dim=0).repeat(self.bs, 1)
            ).float()
            loss_self = F.binary_cross_entropy_with_logits(self_attn, self_label)

            self.optimizer.zero_grad()
            loss = loss_learn + loss_teach + loss_self
            loss.backward()
            self.optimizer.step()

            self._dequeue_and_enqueue(f_sd_k, y_sd)

            loss_avg += loss.item()
            loss_learn_avg += loss_learn.item()
            loss_teach_avg += loss_teach.item()
            loss_self_avg += loss_self.item()

        return {
            "loss": loss_avg / self.num_domains,
            "loss_learn": loss_learn_avg / self.num_domains,
            "loss_teach": loss_teach_avg / self.num_domains,
            "loss_self": loss_self_avg / self.num_domains,
        }

    def attn_score(self, q, k):
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        dim = k.shape[0]

        attn_score = q @ k.T / self.t / (dim**0.5)

        return attn_score

    def predict(self, x):
        all_f = self.K_encoder(x)
        predict = F.softmax(self.classifier(all_f), dim=-1)
        CE = self.causal(all_f, self.queue.clone().detach().T)
        return predict

    def causal(self, all_f, embedding, all_y=None):
        bs = all_f.shape[0]

        # P(Z|X=x')
        attn_score = self.attn_score(all_f, embedding)
        z_x = F.softmax(attn_score, dim=-1)  # [bs, e_num]

        # P(Y=y|Z=z,X=x')
        z_x_ = z_x.unsqueeze(dim=-1).repeat(1, 1, self.num_classes)
        y_zx = self.p_y_zx(all_f=all_f, all_z=embedding)  # [bs*e_num, c_num]

        # ∑ P(Y=y|Z=z,X=x')P(X=x')
        if all_y is None:
            p_x = 1 / bs
            sum_x = (
                (p_x * y_zx.view(bs, -1))
                .view(bs, self.embed_num, self.num_classes)
                .sum(dim=0)
            )  # [e_num, c_num]
        else:
            p_x = self.attn_score(all_f, all_f).sum(1)
            p_x = F.softmax(p_x, dim=0).view(-1, 1)
            sum_x = (
                (p_x * y_zx.view(bs, -1))
                .view(bs, self.embed_num, self.num_classes)
                .sum(dim=0)
            )  # [e_num, c_num]

        # ∑∑ P(Y=y|Z=z,X=x')P(X=x')P(Z|X=x)
        sum_z_ = z_x_ * sum_x.view(1, self.embed_num, self.num_classes).repeat(
            bs, 1, 1
        )  # [bs, e_num, c_num]
        sum_z = sum_z_.sum(dim=1)  # [bs, c_num]
        return sum_z

    def _momentum_update_key_encoder(self,flag):
        """
        Momentum update of the key encoder
        """
        if flag:
            for param_q, param_k in zip(
                self.Q_encoder.parameters(), self.K_encoder.parameters()
            ):
                param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def p_y_zx(self, all_f, all_z):
        bs = all_f.shape[0]
        num = all_z.shape[0]

        all_f = self.head(all_f).unsqueeze(1).repeat(1, num, 1).view(-1, 128)
        all_z = self.head(all_z).unsqueeze(0).repeat(bs, 1, 1).view(-1, 128)
        all_fz = torch.cat((all_f, all_z), dim=-1)

        p_y_zx = F.softmax(self.classifier(self.mixer(all_fz)), dim=-1)

        return p_y_zx  # [bs*e_num, c_num]

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, keys_label):
        # gather keys before updating queue

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.embed_num % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        self.queue_label[ptr : ptr + batch_size] = keys_label
        ptr = (ptr + batch_size) % self.embed_num  # move pointer

        self.queue_ptr[0] = ptr
