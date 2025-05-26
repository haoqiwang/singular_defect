#!/usr/bin/env python
import argparse
import os

from singular_defect import load_llm, pairwise_angles_between
import warnings
import torch
from tqdm import tqdm
from datasets import load_dataset


warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="llama2_7b", help="LLaMA model")
    parser.add_argument("--thr", type=float, default=0, help="high norm threshold")
    parser.add_argument("--access_token", type=str)
    parser.add_argument("--savedir", type=str, default="output")
    parser.add_argument("--pairwise_angle", action="store_true")
    parser.add_argument("--revision", default=None, help="Revision of the model")
    args = parser.parse_args()
    return args


THRESHOLDS = {
    "llama2_7b": 500,
    "llama2_7b_chat": 500,
    "llama2_7b_code": 1000,
    "llama2_13b": 1000,
    "llama2_13b_chat": 1000,
    "llama3_8b": 300,
    "llama3_8b_instruct": 300,
    "llama3_8b_guard": 300,
    "llama3.1_8b": 300,
    "llama3.1_8b_instruct": 300,
    "llama3.1_8b_guard": 300,
    "vicuna1.1_7b": 1000,
    "vicuna1.3_7b": 1000,
    "vicuna1.5_7b": 400,
    "vicuna1.5_7b_16k": 400,
    "qwen2_7b": 6000,
    "qwen2_7b_instruct": 6000,
    "qwen2_7b_math": 8000,
    "qwen2_7b_math_instruct": 8000,
    "qwen2.5_7b": 8000,
    "mpt_7b": 1500,
    "mpt_7b_chat": 1500,
    "mpt_7b_instruct": 1500,
    "mpt_7b_storywriter": 1500,
    "phi3_mini": 2000,
    "phi3_mini_128k": 2000,
    "phi3_medium": 2500,
    "phi3.5_mini": 2000,
    "pythia_160m": 200,
    "pythia_410m_1": 400,
    "pythia_410m_2": 400,
    "falcon2_11b": 3000,
    "gpt2_medium": 3000,
    "qwen2.5_1.5b": 8000,
    "deepseek_r1_llama_8b": 300,
}

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.savedir, exist_ok=True)

    thr = args.thr if args.thr > 0 else THRESHOLDS[args.model]
    print(f"{thr=}")

    model, tokenizer, device, layers, hidden_size, seq_len = load_llm(args)
    print(model)

    dataset = load_dataset(path="wikitext", name="wikitext-2-v1", split="train")
    lines = [line["text"] for line in dataset if line and line["text"] and 10 < len(line["text"]) < 1000][:1000]

    anomalies = []
    low_norms = []
    for seq in tqdm(lines):
        valenc = tokenizer(seq, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        with torch.inference_mode():
            result = model(valenc, output_hidden_states=True)
            feats = torch.cat(result["hidden_states"])
            high_norm_feats = feats[feats.norm(dim=-1) > thr]
            low_norms.append(feats.norm(dim=-1)[feats.norm(dim=-1) < thr])
            if len(high_norm_feats) > 0:
                anomalies.append(high_norm_feats)

    anomaly_feats = torch.cat(anomalies)
    low_norms = torch.cat(low_norms)
    mean_anomaly = anomaly_feats.mean(dim=0)
    print(f"{len(anomaly_feats)=}")
    print(f"{low_norms.mean()=:.2f}")

    post = "" if args.revision is None else f"_{args.revision}"
    path = f"{args.savedir}/{args.model}_sd_empirical{post}.pth"
    print(path)

    torch.save({args.model: {"sds_empirical": mean_anomaly}}, path)

    if args.pairwise_angle:
        angle = pairwise_angles_between(anomaly_feats, anomaly_feats).mean()
        print(f"mean pairwise angle: {angle.item():.2f}")
        torch.save({args.model: {"sds_empirical": mean_anomaly, "pairwise_angle": angle}}, path)
