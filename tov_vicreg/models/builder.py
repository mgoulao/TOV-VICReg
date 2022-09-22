import torch
import torch.nn as nn
import torch.nn.functional as F

from tov_vicreg.models.load_models import load_model


class TOVVICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.log = {}

        self.backbone = load_model(args.arch, patch_size=args.patch_size)
        self.representation_size = self.backbone.num_features
        self.embedding_size = int(args.mlp.split("-")[-1])

        self.projector = Projector(args, self.representation_size)
        self.combinations = torch.tensor([[0, 1, 2],
                                         [0, 2, 1],
                                         [1, 0, 2],
                                         [1, 2, 0],
                                         [2, 0, 1],
                                         [2, 1, 0]])
        self.temporal_order_predictor = nn.Sequential(
            nn.Linear(self.representation_size * 3, 1)
        )
        self.temporal_order_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def covariance_loss(self, cov):
        return off_diagonal(cov).pow_(2).sum().div(
            self.embedding_size
        )

    def forward(self, x, y, z, w):
        # x and y are augmentations of the same source X_t
        # z, w and u are augmentations of a source X_{t-1}, X_{t} and X_{t+1}, respectively

        repr_x  = self.backbone(x)
        repr_y  = self.backbone(y)
        repr_z  = self.backbone(z)
        repr_w  = self.backbone(w)
        # repr_u  = self.backbone(u)

        x = self.projector(repr_x)
        y = self.projector(repr_y)
        z = self.projector(repr_z)
        w = self.projector(repr_w)
        # u = self.projector(repr_u)

        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        z = z - z.mean(dim=0)
        w = w - w.mean(dim=0)
        # u = u - u.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)

        gamma = 1

        # randomly select a combination for each sample
        shuffle_labels = torch.randint(0, self.combinations.shape[0], (self.args.batch_size,))
        # get combination selected
        shuffle_indexes = self.combinations[shuffle_labels].cuda(self.args.gpu)
        # change combination label to binary (correct order or not)
        shuffle_binary_labels = torch.where(shuffle_labels == 0, shuffle_labels, 1).view(-1, 1).type(torch.float).cuda(self.args.gpu)
        # concatenate the representation in correct order
        temporal_samples = torch.cat([repr_x.unsqueeze(1), repr_z.unsqueeze(1), repr_w.unsqueeze(1)], dim=1)

        # Shuffle representations according to the combination selected
        x_1_indexes = shuffle_indexes[:, 0].view(-1, 1).repeat(1, self.representation_size).view(self.args.batch_size, 1, self.representation_size)
        x_1 = torch.gather(temporal_samples, 1, x_1_indexes).squeeze(1)

        x_2_indexes = shuffle_indexes[:, 1].view(-1, 1).repeat(1, self.representation_size).view(self.args.batch_size, 1, self.representation_size)
        x_2 = torch.gather(temporal_samples, 1, x_2_indexes).squeeze(1)

        x_3_indexes = shuffle_indexes[:, 2].view(-1, 1).repeat(1, self.representation_size).view(self.args.batch_size, 1, self.representation_size)
        x_3 = torch.gather(temporal_samples, 1, x_3_indexes).squeeze(1)
        shuffled_concat = torch.concat([x_1, x_2, x_3], dim=1)
        # End of shuffle

        # Predict binary classification
        pred_temporal_class = self.temporal_order_predictor(shuffled_concat)
        # Compute temporal loss
        temporal_loss = self.temporal_order_loss(pred_temporal_class, shuffle_binary_labels)

        std_loss = torch.mean(F.relu(gamma - std_x)) / 2 + \
                    torch.mean(F.relu(gamma - std_y)) / 2

        cov_loss = self.covariance_loss(cov_x) + \
                    self.covariance_loss(cov_y)

        self.log = {
            "temporal_loss": temporal_loss,
            "invariance_loss": repr_loss,
            "variance_loss": std_loss,
            "covariance_loss": cov_loss
        }

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.temporal_coeff * temporal_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
