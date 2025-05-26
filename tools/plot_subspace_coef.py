#!/usr/bin/env python
import argparse
import os

from singular_defect import load_llm, i_am_norm
from matplotlib import pyplot as plt
import matplotlib
import warnings
import torch
import numpy as np
from torch import nn

warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams.update({"figure.max_open_warning": 0})
plt.rcParams["figure.figsize"] = [10, 4]
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="llama2_7b", help="LLaMA model")
    parser.add_argument("--sentence", type=str, default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--access_token", type=str)
    parser.add_argument("--savedir", type=str, default="output")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
    parser.add_argument("--revision", default=None, help="Revision of the model")
    parser.add_argument("--layer_id", default=None, type=int, help="layer to plot")
    parser.add_argument("--no_image", action="store_true", help="Do not save image")
    args = parser.parse_args()
    return args


LAYER_ID_MAP = {
    "llama2_7b": 1,
    "llama2_7b_chat": 1,
    "llama2_7b_code": 1,
    "llama2_13b": 3,
    "llama2_13b_chat": 3,
    "llama3_8b": 1,
    "llama3_8b_instruct": 1,
    "phi3_mini": 2,
    "phi3_mini_128k": 2,
    "phi3_medium": 5,
    "phi3.5_mini": 2,
    "qwen2_7b": 3,
    "qwen2_7b_instruct": 3,
}

LIMIT_MAP = {
    "llama2_7b": [-1.5, 2.5],
    "llama2_7b_chat": [-1.5, 2],
    "llama2_7b_code": [-3.5, 3.5],
    "llama2_13b": [-3.0, 13],
    "llama2_13b_chat": [-3.0, 13],
    "llama3_8b": [-1.5, 3],
    "llama3_8b_instruct": [-1.5, 3],
    "phi3_mini": [-3.5, 9.5],
    "phi3_mini_128k": [-3.5, 9.5],
    "phi3_medium": [-10, 300],
    "phi3.5_mini": [-3.5, 9.5],
    "qwen2_7b": [-5, 11.5],
    "qwen2_7b_instruct": [-5, 11.5],
}

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.savedir, exist_ok=True)
    if args.format == "png":
        matplotlib.use("Agg")
    elif args.format == "pdf":
        matplotlib.use("PDF")
    else:
        raise ValueError("Unsupported format")

    if args.layer_id is None:
        args.layer_id = LAYER_ID_MAP[args.model]

    ylim = LIMIT_MAP.get(args.model, [-1.5, 2.5])

    model, tokenizer, device, layers, hidden_size, seq_len = load_llm(args)
    print(model)

    right_sv = (
        torch.load(f"{args.savedir}/{args.model}_right_sv.pth", weights_only=True)[args.model]["right_sv"]
        .to(device)
        .half()
    )

    inputs = None

    # Define a hook function to capture the output of a specific layer
    def hook_fn(module, input, output):
        global inputs
        inputs = input[0].half()

    # Choose the layer you want to hook
    layer_to_hook = layers[args.layer_id]  # Change the index to the desired layer

    hook = layer_to_hook.post_attention_layernorm.register_forward_hook(hook_fn)

    n_token = max(max(tokenizer.all_special_ids) + 1, tokenizer.vocab_size)
    tokens = model.model.embed_tokens.weight.clone()[:n_token]
    n_token = tokens.size(0)
    avg_norm = tokens.norm(dim=-1).mean()
    trained_mask = tokens.norm(dim=-1) > avg_norm / 10

    batch_size = 5120  # Define your batch size

    # Run a forward pass to trigger the hook in batches
    coefs0_list = []
    for i in range(0, n_token, batch_size):
        input_ids_batch = torch.arange(i, min(i + batch_size, n_token)).unsqueeze(1).to(device)
        with torch.inference_mode():
            model(input_ids_batch)
        coefs0_batch = inputs[:, 0].to(right_sv.device) @ right_sv
        coefs0_batch[~trained_mask[i : i + batch_size]] = np.nan
        coefs0_list.append(coefs0_batch.detach().cpu())

    coefs0 = torch.cat(coefs0_list, dim=0)

    # Run a forward pass with shuffled input_ids in batches
    coefs1_list = []
    shuffled_indices = torch.randperm(n_token)
    for i in range(0, n_token, batch_size):
        shuffled_input_ids_batch = shuffled_indices[i : i + batch_size].unsqueeze(1).to(device)
        input_ids_batch = torch.arange(i, min(i + batch_size, n_token)).unsqueeze(1).to(device)
        with torch.inference_mode():
            model(torch.cat((shuffled_input_ids_batch, input_ids_batch), dim=1))
        coefs1_batch = inputs[:, 1].to(right_sv.device) @ right_sv
        coefs1_batch[~trained_mask[i : i + batch_size]] = np.nan
        coefs1_list.append(coefs1_batch.detach().cpu())

    coefs1 = torch.cat(coefs1_list, dim=0)

    trained_mask = trained_mask.cpu()

    print(
        f"coefs0: {coefs0[trained_mask].mean().item():.1f}+-{coefs0[trained_mask].std():.1f}, coefs1: {coefs1[trained_mask].mean().item():.1f}+-{coefs1[trained_mask].std().item():.1f}"
    )

    plt.scatter(np.arange(n_token) + 1, coefs0, s=2, label="Initial Position", alpha=0.1)
    plt.scatter(np.arange(n_token) + 1, coefs1, s=2, label="Second Position", alpha=0.1)

    leg = plt.legend(loc="center", markerscale=3)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    plt.xlabel("Token ID")
    plt.ylabel("Coefficient")
    plt.ylim(*ylim)
    plt.tight_layout()

    c1_sorted, c1_sorted_index = coefs1.sort()
    mask = c1_sorted.isfinite()
    c1_sorted = c1_sorted[mask]
    c1_sorted_index = c1_sorted_index[mask]
    for i in range(10):
        t = c1_sorted_index[-i - 1].item()
        norm = i_am_norm(t, model, tokenizer, device).max()
        print(f"{i}: {repr(tokenizer.decode(t))} id {t} max norm {norm:.2f} {c1_sorted[-i - 1].item():.2f}")

    if not args.no_image:
        post = "" if args.revision is None else f"_{args.revision}"
        path = f"{args.savedir}/{args.model}_subspace_coef{post}.{args.format}"
        print(path)
        plt.savefig(path)

        if args.format == "png":
            os.system(f"convert {path} -trim {path}")
            os.system(f"pngcrush -reduce -brute -ow {path}")
        elif args.format == "pdf":
            os.system(f"pdfcrop {path} {path}")
