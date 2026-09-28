import torch
import torch.nn as nn
import torch.nn.functional as F


def cos_sim(A, B):
    A_norm = torch.norm(A, dim=1, keepdim=True)
    B_norm = torch.norm(B, dim=1, keepdim=True)
    similarity_matrix = torch.matmul(A, B.T) / (torch.matmul(A_norm, B_norm.T) + 1e-8)
    return similarity_matrix


class NewtonSchulzOrthogonal(nn.Module):
    """
    Differentiable orthogonalization layer (Newton-Schulz Iteration).
    Input: Z (Batch, K)
    Output: H (Batch, K), satisfying H^T H = I
    """

    def __init__(self, iterations=5):
        super(NewtonSchulzOrthogonal, self).__init__()
        self.iterations = iterations

    def forward(self, Z):
        norm = torch.norm(Z, p='fro')

        if norm > 1.5:
            Y = Z / norm * 1.5
        else:
            Y = Z

        batch_size, k = Z.shape
        I = torch.eye(k, device=Z.device)

        for _ in range(self.iterations):
            YtY = torch.matmul(Y.t(), Y)
            Y = 0.5 * torch.matmul(Y, 3 * I - YtY)

        return Y

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential()
        for i in range(len(dims)):
            if i == 0:
                self.encoder.add_module('Linear%d' % i, nn.Linear(input_dim, dims[i]))
            else:
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], dims[i]))
            if i != len(dims) - 1:
                self.encoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        return self.encoder(x)


class DOEN_MultiView(nn.Module):
    def __init__(self, num_views, input_sizes, dims, kernel, num_clusters, batch_size):
        super(DOEN_MultiView, self).__init__()
        self.num_views = num_views
        self.num_clusters = num_clusters
        self.batch_size = batch_size

        self.encoders = nn.ModuleList([
            Encoder(input_sizes[idx], num_clusters, dims) for idx in range(num_views)
        ])
        self.ortho_layer = NewtonSchulzOrthogonal(iterations=5)

        self.label_learning_module = nn.Sequential(
            nn.Linear(dims[len(dims)-1], num_clusters),
        )

        self.sim_fn = ComprehensiveKernel(metric=kernel, t=1.0, mix_weight=0.5)
        self.simWs = None

    def forward(self, data_views):
        Hs, HFs = [], []
        self.simWs = []
        for idx in range(self.num_views):
            high_features = self.encoders[idx](data_views[idx])
            high_features = F.normalize(high_features, p=2, dim=1)
            HFs.append(high_features)
            pairwise_dist = torch.cdist(high_features.detach(), high_features.detach())
            simW = torch.exp(-pairwise_dist ** 2 / (2 * 1 ** 2))
            self.simWs.append(simW)
            z = self.label_learning_module(high_features)

            h = self.ortho_layer(z)
            Hs.append(h)

        h_sum = sum(Hs)
        h_consensus = self.ortho_layer(h_sum)
        HF_sum = sum(HFs)
        pairwise_dist = torch.cdist(HF_sum.detach(), HF_sum.detach())
        simW = torch.exp(-pairwise_dist ** 2 / (2 * 1 ** 2))
        self.simWs.append(simW)

        return Hs, h_consensus

    def loss(self, Hs, h_consensus, adj_graph, temperature_l, alpha, beta):
        """
        alpha: Local spectral cut loss weight (Intra-view Structure)
        beta:  Global spectral alignment loss weight (Inter-view Alignment)
        """

        loss_alignment = self.loss_spectral_fusion(Hs, h_consensus, adj_graph)

        loss_spectral = 0
        for i in range(self.num_views):
            A = self.simWs[i]
            D = torch.sum(A, dim=1) + 1e-5
            D_inv_sqrt = torch.pow(D, -0.5)
            D_mat_inv_sqrt = torch.diag(D_inv_sqrt)
            A_tilde = torch.matmul(torch.matmul(D_mat_inv_sqrt, A), D_mat_inv_sqrt)
            H_single = h_consensus if i == self.num_views else Hs[i]
            trace_val = torch.trace(torch.matmul(torch.matmul(H_single.t(), A_tilde), H_single))
            loss_spectral += -trace_val / self.batch_size

        total_loss = loss_spectral + alpha * loss_alignment

        return total_loss

    def loss_spectral_fusion(self, Hs, F_consensus, adj_graph):
        """
        Spectral embedding contrastive alignment loss (supports hybrid kernels).
        Note: temperature is now managed internally by self.sim_fn.
        """
        positive_graph = adj_graph[0]
        negative_graph = adj_graph[1]

        total_loss = 0

        for i in range(self.num_views):
            F_v = Hs[i]

            sim_vc = self.sim_fn(F_v, F_consensus)

            pos_vc = torch.sum(sim_vc * positive_graph, dim=1) + 1e-8
            neg_vc = torch.sum(sim_vc * negative_graph, dim=1) + 1e-8

            loss_vc = -(torch.log(pos_vc).sum() - torch.log(neg_vc).sum()) / F_v.shape[0]

            total_loss += loss_vc

        return total_loss

class ComprehensiveKernel(nn.Module):
    def __init__(self, metric='cosine', **kwargs):
        """
        Similarity computation module supporting multiple kernel functions.

        Args:
            metric: Kernel function type
                - 'cosine': cosine similarity
                - 'linear': linear kernel (Z * Z^T)
                - 'sigmoid': Sigmoid/Tanh kernel (tanh(a*xy + c))
                - 'polynomial': polynomial kernel ((gamma*xy + r)^d)
                - 'gaussian': Gaussian kernel (RBF)
                - 'laplacian': Laplacian kernel
                - 'imq': Inverse Multiquadric kernel
            kwargs: kernel parameters
                - t (sigma): bandwidth for Gaussian/Laplacian/IMQ
                - degree (d): degree of polynomial kernel
                - gamma, coef0: parameters for polynomial/Sigmoid kernels
        """
        super(ComprehensiveKernel, self).__init__()
        self.metric = metric
        self.kwargs = kwargs

        self.t = kwargs.get('t', 1.0)
        self.degree = kwargs.get('degree', 2.0)
        self.gamma = kwargs.get('gamma', 1.0)
        self.coef0 = kwargs.get('coef0', 1.0)

    def forward(self, x, y):
        """
        Input: x [N, D], y [N, D]
        Output: Similarity Matrix [N, N]
        """
        if self.metric in ['cosine', 'linear', 'polynomial', 'sigmoid']:
            x = F.normalize(x, p=2, dim=1)
            y = F.normalize(y, p=2, dim=1)

        inner_prod = torch.matmul(x, y.T)

        if self.metric == 'cosine':
            return torch.exp(inner_prod / self.t)

        elif self.metric == 'linear':
            return inner_prod

        elif self.metric == 'sigmoid':
            return torch.tanh(self.gamma * inner_prod + self.coef0)

        elif self.metric == 'polynomial':
            return torch.pow(self.gamma * inner_prod + self.coef0, self.degree)

        else:
            dist_sq = torch.cdist(x, y, p=2) ** 2

            if self.metric == 'gaussian':
                return torch.exp(-dist_sq / (2 * self.t ** 2))

            elif self.metric == 'laplacian':
                dist = torch.sqrt(dist_sq + 1e-8)
                return torch.exp(-dist / self.t)

            elif self.metric == 'imq':
                return 1.0 / torch.sqrt(dist_sq + self.t ** 2)

        raise ValueError(f"Unknown metric: {self.metric}")

    def extra_repr(self):
        return f"metric={self.metric}, params={self.kwargs}"
