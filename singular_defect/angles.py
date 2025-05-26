import torch


@torch.no_grad()
def angles_between(a, b):
    a = a.to(b.device)
    a = a.double()
    b = b.double()
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return torch.rad2deg((a * b).sum(dim=-1).detach().cpu().clamp(-1, 1).arccos())


@torch.no_grad()
def acute_angles_between(a, b):
    return 90 - (angles_between(a, b) - 90).abs()


@torch.no_grad()
def pairwise_angles_between(a, b):
    assert a.ndim == 2 and b.ndim == 2
    return torch.stack([acute_angles_between(b, aa) for aa in a])
