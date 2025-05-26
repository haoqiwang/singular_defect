import torch


@torch.no_grad()
def extract_feats(seq, model, tokenizer, device, return_ids=False):
    if isinstance(seq, list):
        ids = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    else:
        ids = tokenizer(seq, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    result = model(ids, output_hidden_states=True)
    feats = torch.cat(result["hidden_states"])

    if return_ids:
        return feats, ids
    return feats


@torch.no_grad()
def i_am_norm(token, model, tokenizer, device):
    pre_tokens = tokenizer("I am", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    seq = list(torch.cat((pre_tokens, torch.tensor(token).reshape(-1)), dim=0))
    feats = extract_feats(seq, model, tokenizer, device)
    norms = feats.norm(dim=-1)[:, -1]
    return norms
