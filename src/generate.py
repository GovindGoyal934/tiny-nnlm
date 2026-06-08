import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
import torch.nn.functional as F
import config
from model import NNLM
from dataset import load_data, tokenize


def _candidate_checkpoint_paths():
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(script_dir)
    return [
        os.path.join(script_dir, "nnlm_best.pt"),
        os.path.join(script_dir, "nnlm.pt"),
        os.path.join(project_root, "nnlm_best.pt"),
        os.path.join(project_root, "nnlm.pt"),
    ]


def _unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            maybe_state_dict = checkpoint.get(key)
            if isinstance(maybe_state_dict, dict):
                return maybe_state_dict
    return checkpoint


def _state_dict_shapes(state_dict):
    return {
        name: tuple(param.shape)
        for name, param in state_dict.items()
        if hasattr(param, "shape")
    }


def _load_compatible_checkpoint(model):
    expected_shapes = _state_dict_shapes(model.state_dict())
    candidates = []

    for path in _candidate_checkpoint_paths():
        if not os.path.exists(path):
            continue

        checkpoint = _unwrap_state_dict(torch.load(path, map_location=config.DEVICE))
        if not isinstance(checkpoint, dict):
            continue

        checkpoint_shapes = _state_dict_shapes(checkpoint)
        if checkpoint_shapes == expected_shapes:
            candidates.append((os.path.getmtime(path), path, checkpoint))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, path, checkpoint = candidates[0]
    model.load_state_dict(checkpoint)
    return path, checkpoint


def top_k_filter(logits, top_k):
    if top_k is None or top_k <= 0:
        return logits

    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    cutoff = values[..., -1, None]
    filtered = logits.masked_fill(logits < cutoff, float("-inf"))
    return filtered


def top_p_filter(logits, top_p):
    if top_p is None or top_p <= 0 or top_p >= 1:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered


@torch.no_grad()
def generate(seed_text, max_length=20, temperature=0.7):
    _, _, _, vocab = load_data()
    model = NNLM(vocab_size=len(vocab)).to(config.DEVICE)
    # prepare pad id so we can prevent generating pad tokens
    itos_tmp = vocab.get_itos()
    stoi = {tok: i for i, tok in enumerate(itos_tmp)}
    pad_id = stoi.get("<pad>", stoi.get("<unk>", 0))
    ckpt_path, ckpt = _load_compatible_checkpoint(model)
    if ckpt_path is None:
        expected_shapes = _state_dict_shapes(model.state_dict())
        msg_lines = [
            "No checkpoint matched the current model architecture.",
            "Expected parameter shapes:",
        ]
        for name, shape in expected_shapes.items():
            msg_lines.append(f"  {name}: {shape}")
        msg_lines.append("\nChecked checkpoint files:")
        found_any = False
        for path in _candidate_checkpoint_paths():
            if not os.path.exists(path):
                continue
            found_any = True
            checkpoint = _unwrap_state_dict(torch.load(path, map_location=config.DEVICE))
            if isinstance(checkpoint, dict):
                msg_lines.append(f"  {path}")
                for name, shape in _state_dict_shapes(checkpoint).items():
                    msg_lines.append(f"    {name}: {shape}")
        if not found_any:
            msg_lines.append("  No checkpoint files were found.")
        msg_lines.append(
            "\nUse a checkpoint trained with the current config, or retrain with matching hyperparameters."
        )
        raise RuntimeError("\n".join(msg_lines))
    model.eval()

    tokens = vocab(tokenize(seed_text))
    if len(tokens) < config.CONTEXT_WINDOW:
        # torchtext Vocab may not expose a stoi; build one from itos()
        itos_tmp = vocab.get_itos()
        stoi = {tok: i for i, tok in enumerate(itos_tmp)}
        pad_id = stoi.get("<pad>", stoi.get("<unk>", 0))
        tokens = [pad_id] * (config.CONTEXT_WINDOW - len(tokens)) + tokens

    for _ in range(max_length):
        context = torch.tensor(
            tokens[-config.CONTEXT_WINDOW :], dtype=torch.long
        ).unsqueeze(0).to(config.DEVICE)
        logits = model(context)
        logits = logits / temperature
        # never sample the padding token
        if 0 <= pad_id < logits.size(-1):
            logits[..., pad_id] = float("-inf")
        logits = top_p_filter(logits, top_p=getattr(config, "TOP_P", 0.9))
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        tokens.append(next_token)

    itos = vocab.get_itos()
    itos = vocab.get_itos()
    words = [itos[t] for t in tokens if t != pad_id]
    return " ".join(words)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text with trained NNLM"
    )
    parser.add_argument(
        "--seed", type=str, default="he portrayed"
    )
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    result = generate(args.seed, args.max_length, args.temperature)
    print(result)
