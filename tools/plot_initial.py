#!/usr/bin/env python
import argparse
import os

from singular_defect import load_llm
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

    model, tokenizer, device, layers, hidden_size, seq_len = load_llm(args)
    print(model)

    layer_output = []

    # Define a hook function to capture the output of a specific layer
    def hook_fn(module, input, output):
        layer_output.append(output[0])

    # Choose the layer you want to hook
    layer_to_hook = layers[args.layer_id]  # Change the index to the desired layer

    hook = layer_to_hook.register_forward_hook(hook_fn)

    n_token = max(max(tokenizer.all_special_ids) + 1, tokenizer.vocab_size)
    tokens = model.model.embed_tokens.weight.clone()[:n_token]
    n_token = tokens.size(0)
    avg_norm = tokens.norm(dim=-1).mean()
    trained_mask = tokens.norm(dim=-1) > avg_norm / 10

    # Run a forward pass to trigger the hook
    input_ids = torch.arange(n_token).unsqueeze(1).to(device)
    bs = 8192
    with torch.inference_mode():
        for i in range(0, n_token, bs):
            input_ids = torch.arange(i, min(i + bs, n_token)).unsqueeze(1).to(device)
            model(input_ids)

    trained_output = torch.cat(layer_output, dim=0)
    # The output feature of the hooked layer is now stored in `layer_output`
    print(trained_output.shape)

    layer_output.clear()
    with torch.inference_mode():
        model.model.embed_tokens.weight.data = (
            nn.functional.normalize(torch.randn_like(model.model.embed_tokens.weight.data), dim=-1) * avg_norm
        )
        for i in range(0, n_token, bs):
            input_ids = torch.arange(i, min(i + bs, n_token)).unsqueeze(1).to(device)
            model(input_ids)

    random_output = torch.cat(layer_output, dim=0)

    norms_sorted, ids_sorted = trained_output.norm(dim=-1).flatten().detach().cpu().sort()
    norms = trained_output.norm(dim=-1).flatten().detach().cpu()
    norms[~trained_mask] = np.nan
    plt.scatter(np.arange(n_token), norms, s=4, label="Trained Tokens", alpha=0.5)

    plt.xlabel("Token ID")
    plt.ylabel(f"Norm after Layer {args.layer_id + 1}")

    plt.scatter(
        np.arange(n_token),
        random_output.norm(dim=-1).flatten().sort()[0].detach().cpu(),
        s=4,
        marker="x",
        label="Random Embeddings (sorted)",
    )

    plt.legend(loc="center")
    plt.tight_layout()

    post = "" if args.revision is None else f"_{args.revision}"
    path = f"{args.savedir}/{args.model}_initial{post}.{args.format}"
    print(path)
    plt.savefig(path)

    if args.format == "png":
        os.system(f"convert {path} -trim {path}")
        os.system(f"pngcrush -reduce -brute -ow {path}")
    elif args.format == "pdf":
        os.system(f"pdfcrop {path} {path}")
