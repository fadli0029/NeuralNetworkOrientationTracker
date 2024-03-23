import torch

EPS = 1e-6

def qmult_pytorch(q1, q2):
    """
    Batched version of quaternion multiplication using PyTorch.

    Args:
        q1: torch.Tensor, shape (..., 4)
        q2: torch.Tensor, shape (..., 4)

    Returns:
        torch.Tensor, shape (..., 4)
    """
    # Extract scalar and vector components
    q1s, q1v = q1[..., :1], q1[..., 1:]
    q2s, q2v = q2[..., :1], q2[..., 1:]

    # Compute scalar and vector components of the product
    q3s = q1s * q2s - torch.sum(q1v * q2v, dim=-1, keepdim=True)
    q3v = q1s * q2v + q2s * q1v + torch.cross(q1v, q2v, dim=-1)

    # Combine and return
    return torch.cat([q3s, q3v], dim=-1)

def qinverse_pytorch(q):
    """
    Batched version of quaternion inverse using PyTorch.

    Args:
        q: torch.Tensor, shape (..., 4)

    Returns:
        torch.Tensor, shape (..., 4)
    """
    q_conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    q_norm_sq = torch.sum(q ** 2, dim=-1, keepdim=True) + EPS
    return q_conj / q_norm_sq

def qexp_pytorch(q):
    """
    Batched version of quaternion exponential using PyTorch.

    Args:
        q: torch.Tensor, shape (..., 4)

    Returns:
        torch.Tensor, shape (..., 4)
    """
    qv = q[..., 1:]
    qv_norm = torch.norm(qv, dim=-1, keepdim=True) + EPS
    qv_normed = qv / qv_norm
    qv_scaled = qv_normed * torch.sin(qv_norm)
    q_scalar_exp = torch.exp(q[..., :1])
    q_term = torch.cat([torch.cos(qv_norm), qv_scaled], dim=-1)
    return q_scalar_exp * q_term

def qlog_pytorch(q):
    """
    Batched version of quaternion logarithm using PyTorch.

    Args:
        q: torch.Tensor, shape (..., 4)

    Returns:
        torch.Tensor, shape (..., 4)
    """
    qv = q[..., 1:]
    qv_norm = torch.norm(qv, dim=-1, keepdim=True) + EPS
    qv_normed = qv / qv_norm
    q_norm = torch.norm(q, dim=-1, keepdim=True) + EPS
    q_scalar_log = torch.log(q_norm)
    qv_scaled = qv_normed * torch.arccos(q[..., :1] / q_norm)
    return torch.cat([q_scalar_log, qv_scaled], dim=-1)

