import torch
import torch.nn.functional as F
from tqdm import tqdm


def svd(A):
    dev = A.device
    if dev.type == "cpu":
        A = A.cuda()
    u, s, vt = torch.linalg.svd(A)
    return u.to(dev), s.to(dev), vt.to(dev)


# region llama2


@torch.no_grad()
def singular_dir_attn_llama2(blk, identity=False):
    N = blk.input_layernorm.weight.shape[0]
    dev = blk.input_layernorm.weight.device

    A3 = blk.self_attn.o_proj.weight.float()
    A2 = blk.self_attn.v_proj.weight.float()
    r, c = A2.shape
    if r < c:
        # llama3 repeats k, v
        # 1024 x 4096 -> (8 x 128) x 4096 -> 8 x 4 x 128 x 4096 -> 4096 x 4096
        num_heads = getattr(blk.self_attn, "num_heads", blk.self_attn.config.num_attention_heads)
        head_dim = getattr(blk.self_attn, "head_dim", c // num_heads)
        n_repeat = blk.self_attn.num_key_value_groups
        n_kvhead = getattr(blk.self_attn, "num_key_value_heads", num_heads // n_repeat)
        assert r * n_repeat == c
        assert n_kvhead * n_repeat == num_heads
        A2 = A2.view(n_kvhead, head_dim, c)[:, None].expand(-1, n_repeat, -1, -1).reshape(-1, c)
    A1 = torch.diag(blk.input_layernorm.weight.float())
    A = A3 @ A2 @ A1

    if identity:
        A += torch.eye(N).to(dev)
    u, s, vt = svd(A)

    return A, u, s, vt


@torch.no_grad()
def w12_llama2(blk, x):
    dtype = blk.mlp.gate_proj.weight.dtype
    x1 = blk.mlp.gate_proj(x.to(dtype))
    x2 = blk.mlp.up_proj(x.to(dtype))
    return F.silu(x1) * x2


@torch.no_grad()
def singular_dir_mlp_llama2(blk, identity=False, layer_norm=True):
    N = blk.post_attention_layernorm.weight.shape[0]
    dev = blk.post_attention_layernorm.weight.device

    A3 = blk.mlp.down_proj.weight.float()

    X = torch.randn(100000, N, device=dev)
    Y = w12_llama2(blk, X).float()
    X_one = X
    sol = torch.linalg.lstsq(X_one, Y)
    A2 = sol.solution.T.float()

    A1 = torch.diag(blk.post_attention_layernorm.weight.float())
    if layer_norm:
        A = A3 @ A2 @ A1
    else:
        A = A3 @ A2

    if identity:
        A += torch.eye(N).to(dev)
    u, s, vt = svd(A)

    return A, u, s, vt


@torch.no_grad()
def singular_dir_layer_llama2(blk):
    A_attn, u_attn, s_attn, vt_attn = singular_dir_attn_llama2(blk, identity=True)
    A_mlp, u_mlp, s_mlp, vt_mlp = singular_dir_mlp_llama2(blk, identity=True)

    A = A_mlp @ A_attn
    u, s, vt = svd(A)

    return dict(
        A_attn=A_attn.detach().cpu(),
        u0_attn=u_attn[:, 0].half().detach().cpu(),
        s_attn=s_attn.half().detach().cpu(),
        v0_attn=vt_attn[0].half().detach().cpu(),
        A_mlp=A_mlp.detach().cpu(),
        u0_mlp=u_mlp[:, 0].half().detach().cpu(),
        s_mlp=s_mlp.half().detach().cpu(),
        v0_mlp=vt_mlp[0].half().detach().cpu(),
        A=A.detach().cpu(),
        u0=u[:, 0].half().detach().cpu(),
        s=s.half().detach().cpu(),
        v0=vt[0].half().detach().cpu(),
    )


# endregion


# region phi3


@torch.no_grad()
def singular_dir_attn_phi3(blk, identity=False):
    N = blk.input_layernorm.weight.shape[0]
    dev = blk.input_layernorm.weight.device

    A3 = blk.self_attn.o_proj.weight.float()

    n_repeat = blk.self_attn.num_key_value_groups
    A2 = blk.self_attn.qkv_proj.weight.float()
    _, c = A2.shape

    if n_repeat == 1:
        A2 = A2.chunk(3, dim=0)[-1]
    else:
        # phi3 medium repeats k, v
        # 1280 x 5120 -> (10 x 128) x 5120 -> 10 x 4 x 128 x 5120 -> 5120 x 5120
        num_heads = getattr(blk.self_attn, "num_heads", blk.self_attn.config.num_attention_heads)
        head_dim = getattr(blk.self_attn, "head_dim", c // num_heads)
        n_kvhead = getattr(blk.self_attn, "num_key_value_heads", num_heads // n_repeat)
        assert n_kvhead * n_repeat == num_heads

        r = c // n_repeat
        A2 = A2[-r:]
        assert r * n_repeat == c
        A2 = A2.view(n_kvhead, head_dim, c)[:, None].expand(-1, n_repeat, -1, -1).reshape(-1, c)

    A1 = torch.diag(blk.input_layernorm.weight.float())
    A = A3 @ A2 @ A1

    if identity:
        A += torch.eye(N).to(dev)
    u, s, vt = svd(A)

    return A, u, s, vt


@torch.no_grad()
def w12_phi3(blk, x):
    dtype = blk.mlp.gate_up_proj.weight.dtype
    x1, x2 = blk.mlp.gate_up_proj(x.to(dtype)).chunk(2, dim=-1)
    return F.silu(x1) * x2


@torch.no_grad()
def singular_dir_mlp_phi3(blk, identity=False):
    N = blk.post_attention_layernorm.weight.shape[0]
    dev = blk.post_attention_layernorm.weight.device

    A3 = blk.mlp.down_proj.weight.float()

    X = torch.randn(100000, N, device=dev)
    Y = w12_phi3(blk, X).float()
    X_one = X
    sol = torch.linalg.lstsq(X_one, Y)
    A2 = sol.solution.T.float()

    A1 = torch.diag(blk.post_attention_layernorm.weight.float())
    A = A3 @ A2 @ A1

    if identity:
        A += torch.eye(N).to(dev)
    u, s, vt = svd(A)

    return A, u, s, vt


@torch.no_grad()
def singular_dir_layer_phi3(blk):
    A_attn, u_attn, s_attn, vt_attn = singular_dir_attn_phi3(blk, identity=True)
    A_mlp, u_mlp, s_mlp, vt_mlp = singular_dir_mlp_phi3(blk, identity=True)

    A = A_mlp @ A_attn
    u, s, vt = svd(A)

    return dict(
        A_attn=A_attn.detach().cpu(),
        u0_attn=u_attn[:, 0].half().detach().cpu(),
        s_attn=s_attn.half().detach().cpu(),
        v0_attn=vt_attn[0].half().detach().cpu(),
        A_mlp=A_mlp.detach().cpu(),
        u0_mlp=u_mlp[:, 0].half().detach().cpu(),
        s_mlp=s_mlp.half().detach().cpu(),
        v0_mlp=vt_mlp[0].half().detach().cpu(),
        A=A.detach().cpu(),
        u0=u[:, 0].half().detach().cpu(),
        s=s.half().detach().cpu(),
        v0=vt[0].half().detach().cpu(),
    )


# endregion


# region pythia


@torch.no_grad()
def singular_dir_attn_pythia(blk, identity=False):
    N = blk.input_layernorm.weight.shape[0]
    dev = blk.input_layernorm.weight.device

    A3 = blk.attention.dense.weight.float()

    n_heads = blk.attention.num_attention_heads
    head_size = blk.attention.head_size
    A2 = blk.attention.query_key_value.weight
    A2 = A2.reshape(n_heads, 3, head_size, A2.size(1))[:, -1].reshape(-1, A2.size(1)).float()

    A1 = torch.diag(blk.input_layernorm.weight).float()
    A = A3 @ A2 @ A1

    if identity:
        A += torch.eye(N).to(dev)
    u, s, vt = torch.linalg.svd(A)

    return A, u, s, vt


@torch.no_grad()
def w12_pythia(blk, x):
    dtype = blk.mlp.dense_h_to_4h.weight.dtype
    return blk.mlp.act(blk.mlp.dense_h_to_4h(x.to(dtype)))


@torch.no_grad()
def singular_dir_mlp_pythia(blk, identity=False):
    with torch.no_grad():
        N = blk.post_attention_layernorm.weight.shape[0]
        dev = blk.post_attention_layernorm.weight.device

        A3 = blk.mlp.dense_4h_to_h.weight.float()

        X = torch.randn(100000, N, device=dev)
        Y = w12_pythia(blk, X).float()
        X_one = X
        sol = torch.linalg.lstsq(X_one, Y)
        A2 = sol.solution.T.float()

        A1 = torch.diag(blk.post_attention_layernorm.weight.float())
        A = A3 @ A2 @ A1

        if identity:
            A += torch.eye(N).to(dev)
        u, s, vt = torch.linalg.svd(A)

    return A, u, s, vt


@torch.no_grad()
def singular_dir_layer_pythia(blk):
    assert blk.use_parallel_residual
    A_attn, u_attn, s_attn, vt_attn = singular_dir_attn_pythia(blk, identity=True)
    A_mlp, u_mlp, s_mlp, vt_mlp = singular_dir_mlp_pythia(blk, identity=True)

    A = A_mlp + A_attn - torch.eye(A_mlp.size(0)).to(A_mlp.device)
    u, s, vt = torch.linalg.svd(A)

    return dict(
        A_attn=A_attn.detach().cpu(),
        u0_attn=u_attn[:, 0].half().detach().cpu(),
        s_attn=s_attn.half().detach().cpu(),
        v0_attn=vt_attn[0].half().detach().cpu(),
        A_mlp=A_mlp.detach().cpu(),
        u0_mlp=u_mlp[:, 0].half().detach().cpu(),
        s_mlp=s_mlp.half().detach().cpu(),
        v0_mlp=vt_mlp[0].half().detach().cpu(),
        A=A.detach().cpu(),
        u0=u[:, 0].half().detach().cpu(),
        s=s.half().detach().cpu(),
        v0=vt[0].half().detach().cpu(),
    )


# endregion

# region mpt


@torch.no_grad()
def singular_dir_attn_mpt(blk, identity=False):
    N = blk.norm_1.weight.shape[0]
    dev = blk.norm_1.weight.device

    A3 = blk.attn.out_proj.weight.float()
    A2 = blk.attn.Wqkv.weight[-(blk.attn.kv_n_heads * blk.attn.head_dim) :].float()
    r, c = A2.shape
    assert r == c
    A1 = torch.diag(blk.norm_1.weight.float())
    A = A3 @ A2 @ A1

    if identity:
        A += torch.eye(N).to(dev)
    u, s, vt = svd(A)

    return A, u, s, vt


@torch.no_grad()
def w12_mpt(blk, x):
    dtype = blk.ffn.up_proj.weight.dtype
    return blk.ffn.act(blk.ffn.up_proj(x.to(dtype)))


@torch.no_grad()
def singular_dir_mlp_mpt(blk, identity=False):
    N = blk.norm_2.weight.shape[0]
    dev = blk.norm_2.weight.device

    A3 = blk.ffn.down_proj.weight.float()

    X = torch.randn(100000, N, device=dev)
    Y = w12_mpt(blk, X).float()
    X_one = X
    sol = torch.linalg.lstsq(X_one, Y)
    A2 = sol.solution.T.float()

    A1 = torch.diag(blk.norm_2.weight.float())
    A = A3 @ A2 @ A1

    if identity:
        A += torch.eye(N).to(dev)
    u, s, vt = svd(A)

    return A, u, s, vt


@torch.no_grad()
def singular_dir_layer_mpt(blk):
    A_attn, u_attn, s_attn, vt_attn = singular_dir_attn_mpt(blk, identity=True)
    A_mlp, u_mlp, s_mlp, vt_mlp = singular_dir_mlp_mpt(blk, identity=True)

    A = A_mlp @ A_attn
    u, s, vt = svd(A)

    return dict(
        A_attn=A_attn.detach().cpu(),
        u0_attn=u_attn[:, 0].half().detach().cpu(),
        s_attn=s_attn.half().detach().cpu(),
        v0_attn=vt_attn[0].half().detach().cpu(),
        A_mlp=A_mlp.detach().cpu(),
        u0_mlp=u_mlp[:, 0].half().detach().cpu(),
        s_mlp=s_mlp.half().detach().cpu(),
        v0_mlp=vt_mlp[0].half().detach().cpu(),
        A=A.detach().cpu(),
        u0=u[:, 0].half().detach().cpu(),
        s=s.half().detach().cpu(),
        v0=vt[0].half().detach().cpu(),
    )


# endregion


def layerwise_singular_dir(model_name, layers, return_A=False):
    if "llama2" in model_name:
        singular_dir_layer = singular_dir_layer_llama2
    elif "llama3" in model_name:
        singular_dir_layer = singular_dir_layer_llama2
    elif "phi3" in model_name:
        singular_dir_layer = singular_dir_layer_phi3
    elif "pythia" in model_name:
        singular_dir_layer = singular_dir_layer_pythia
    elif "vicuna" in model_name:
        singular_dir_layer = singular_dir_layer_llama2
    elif "qwen" in model_name:
        singular_dir_layer = singular_dir_layer_llama2
    elif "mpt" in model_name:
        singular_dir_layer = singular_dir_layer_mpt
    elif "falcon" in model_name:
        singular_dir_layer = singular_dir_layer_llama2
    else:
        raise NotImplementedError

    def clean(d):
        if not return_A:
            del d["A"]
            del d["A_attn"]
            del d["A_mlp"]
        return d

    return [clean(singular_dir_layer(layer)) for layer in tqdm(layers)]
