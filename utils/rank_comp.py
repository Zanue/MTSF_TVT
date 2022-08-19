import torch

def l1_matrix_norm(x):
    return x.abs().sum(dim=-2).max(dim=-1).values

def linf_matrix_norm(x):
    return l1_matrix_norm(x.transpose(-2, -1))

def composite_norm(x):
    return torch.sqrt(l1_matrix_norm(x) * linf_matrix_norm(x))

def cal_ratios(x): #[B, N, E]
    residuals = x - x.mean(dim=-2, keepdim=True)
    ratios = composite_norm(residuals) / composite_norm(x)

    return ratios