#!/usr/bin/env python
import argparse
import os

from singular_defect import layerwise_singular_dir, load_llm, acute_angles_between
import numpy as np
import warnings
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="llama2_7b", help="LLaMA model")
    parser.add_argument("--access_token", type=str)
    parser.add_argument("--savedir", type=str, default="output")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.savedir, exist_ok=True)

    sd_empirical = torch.load(f"{args.savedir}/{args.model}_sd_empirical.pth", weights_only=True)[args.model][
        "sds_empirical"
    ]

    model, tokenizer, device, layers, hidden_size, seq_len = load_llm(args)
    print(model)

    results = layerwise_singular_dir(args.model, layers, return_A=True)
    mats = [r["A"] for r in results]

    eig_angles = []
    eig_vals = []
    eig_vecs = []
    for i, mat in tqdm(enumerate(mats)):
        eig_, eigvec_ = torch.linalg.eig(mat.float() - torch.eye(mat.shape[0]).to(mat))
        angles_ = acute_angles_between(sd_empirical, eigvec_.real.T)
        idx = angles_.sort()[1]
        eig_angles.append(angles_[idx])
        eig_vals.append(eig_[idx].real)
        eig_vecs.append(eigvec_.real.T[idx[0]])

        print(f"{i}, {eig_.real[idx[:10]]=}")
        print(f"{i}, {angles_[idx[:10]]=}")

    eig_angles = torch.stack(eig_angles)
    eig_vals = torch.stack(eig_vals)
    eig_vecs = torch.stack(eig_vecs)

    path = f"{args.savedir}/{args.model}_eig.pth"
    print(path)

    torch.save(
        {
            args.model: {
                "eig_angles": eig_angles,
                "eig_vals": eig_vals,
                "eig_vecs": eig_vecs,
            }
        },
        path,
    )
