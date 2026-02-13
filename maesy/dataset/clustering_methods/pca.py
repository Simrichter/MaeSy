import torch
from torch.linalg import matmul


def pca_reduction(data, desired_dim):
    q = min(desired_dim, data.shape[0]-1)
    if q < 256:
        print(f"Warning: Not enough samples for accurate PCA reduction. Reducing to {q} dimensions instead of 256.")

    U, S, V = torch.pca_lowrank(data, center=True, q=q)
    out_pca = matmul(data, V)
    return out_pca