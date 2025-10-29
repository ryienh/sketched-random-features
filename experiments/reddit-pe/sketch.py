import torch
import numpy as np
import torch_geometric

torch_geometric.seed_everything(442)


class AddFeaturesTransform:
    def __init__(
        self,
        D_out=16,  # final feature dimension after k sketches
        k=1,  # number of AG sketches to concatenate
        max_size=4000,
        random_state=442,
    ):
        """
        :param D_out: Total output dimension of the final sketch.
        :param k: Number of AG sketches to concatenate. Each sketch has dimension D_in = D_out // k.
        :param alpha: Scalar for the identity matrix in AG sketch operator (alpha * I).
        :param beta: Scalar factor for the Gaussian component in AG sketch operator.
        :param max_size: Maximum number of nodes; used for constructing the largest needed matrices.
        :param random_state: Seed for reproducibility.
        """
        self.D_out = D_out
        self.k = k
        self.D_in = D_out // k  # dimension per sketch

        self.max_size = max_size

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # For each of the k sketches, build:
        #   - A separate AG matrix S_global^i = alpha * I + (beta / sqrt(N)) * G^i
        #   - Random projection matrices for RBF, Laplace, etc.
        self.S_list = []  # list of (N x N) AG matrices
        self.R_linear_list = []  # linear kernel approximation
        self.W_rbf_list = []  # RBF random weights
        self.b_rbf_list = []  # RBF random bias
        self.W_laplace_list = []  # Laplace random weights
        self.b_laplace_list = []  # Laplace random bias

        # Construct each AG sketch + parameter set
        for _ in range(k):
            G = torch.randn(max_size, max_size)
            S_i = 1 * torch.eye(max_size) + (1 / np.sqrt(max_size)) * G
            self.S_list.append(S_i)

            # For a linear kernel, we might approximate with R_in in [F x D_in]
            # We'll do [max_size x D_in] then slice as needed.
            R_lin_i = torch.randn(max_size, self.D_in) / np.sqrt(self.D_in)
            self.R_linear_list.append(R_lin_i)

            # RBF random Fourier features
            W_rbf_i = torch.randn(self.D_in, max_size)
            b_rbf_i = torch.rand(self.D_in) * (2.0 * np.pi)
            self.W_rbf_list.append(W_rbf_i)
            self.b_rbf_list.append(b_rbf_i)

            # Laplace (Cauchy) random features
            W_laplace_i = torch.tensor(
                np.random.standard_cauchy(size=(self.D_in, max_size)),
                dtype=torch.float32,
            )
            b_laplace_i = torch.rand(self.D_in) * (2.0 * np.pi)
            self.W_laplace_list.append(W_laplace_i)
            self.b_laplace_list.append(b_laplace_i)

    def __call__(self, data):
        return self.add_features(data)

    def add_features(self, data):
        """
        For each graph:
          1) Possibly convert data.x to float.
          2) Compute random Fourier features (RBF, Laplace), linear features, etc.
          3) For each of the k sketches, slice and multiply to get the final embedding chunk.
          4) Concatenate the k chunks to form the final embedding of dimension D_out.
        """
        if data.x is None:
            # default node features if none
            data.x = torch.ones((data.num_nodes, 4))

        convert_back = False
        if data.x.dtype == torch.int64:
            data.x = data.x.to(torch.float32)
            convert_back = True

        N, F = data.x.shape

        # We'll store partial sketches in a list, then cat them along dim=1
        rbf_parts = []
        linear_parts = []
        laplace_parts = []

        for i in range(self.k):
            # slice out the NxN, NxF for current graph
            S_i = self.S_list[i][:N, :N]

            # Linear features for chunk i
            R_lin_i = self.R_linear_list[i][:F, :]  # shape [F, D_in]
            linear_approx = data.x @ R_lin_i  # shape [N, D_in]
            linear_sketch = S_i @ linear_approx  # shape [N, D_in]
            linear_parts.append(linear_sketch)

            # RBF features for chunk i
            W_rbf_i = self.W_rbf_list[i][:, :F]  # shape [D_in, F]
            b_rbf_i = self.b_rbf_list[i]  # shape [D_in]
            # WX -> shape [N, D_in] after transpose
            WX_rbf = data.x @ W_rbf_i.T
            # Standard RFF scaling
            Z_rbf = torch.sqrt(torch.tensor(2.0 / self.D_in)) * torch.cos(
                WX_rbf + b_rbf_i
            )
            rbf_sketch = S_i @ Z_rbf
            rbf_parts.append(rbf_sketch)

            # Laplace (Cauchy) features for chunk i
            W_laplace_i = self.W_laplace_list[i][:, :F]  # shape [D_in, F]
            b_laplace_i = self.b_laplace_list[i]  # shape [D_in]
            WX_laplace = data.x @ W_laplace_i.T
            Z_laplace = torch.sqrt(torch.tensor(2.0 / self.D_in)) * torch.cos(
                WX_laplace + b_laplace_i
            )
            laplace_sketch = S_i @ Z_laplace
            laplace_parts.append(laplace_sketch)

        # Now concatenate each partial chunk along dim=1 => shape [N, k * D_in = D_out]
        data.rbf_feats = torch.cat(rbf_parts, dim=1)
        data.linear_feats = torch.cat(linear_parts, dim=1)
        data.laplace_feats = torch.cat(laplace_parts, dim=1)

        # Provide some random features for comparison
        data.random_feats = torch.rand_like(data.rbf_feats)

        # Example: store a no-sketch Laplace for ablations
        data.laplace_feats_no_sketch = torch.cat(
            [
                torch.sqrt(torch.tensor(2.0 / self.D_in))
                * torch.cos(
                    (data.x @ self.W_laplace_list[i][:, :F].T) + self.b_laplace_list[i]
                )
                for i in range(self.k)
            ],
            dim=1,
        )

        if convert_back:
            data.x = data.x.to(torch.int64)

        return data


if __name__ == "__main__":
    from torch_geometric.datasets import ZINC

    # Example usage
    transform = AddFeaturesTransform(D_out=16, k=2, alpha=0.8, beta=0.6, max_size=1000)
    dataset = ZINC(root="data/zinc", subset=True, split="train", transform=transform)

    # Check the shape of the new features
    sample = dataset[0]
    print("Num nodes:", sample.num_nodes)
    print("rbf_feats shape:", sample.rbf_feats.shape)  # Should be [N, 16]
    print("linear_feats shape:", sample.linear_feats.shape)  # Should be [N, 16]
    print("laplace_feats shape:", sample.laplace_feats.shape)
