import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import json
import random
import numpy as np


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def apply_topk_mask(tensor, ratio=0.5): #ratio:keep 50% 的 neuron
    k = max(1, int(tensor.size(-1) * ratio))
    score = tensor.abs()
    topk_idx = torch.topk(score, k, dim=-1).indices #get top-k 個neuron
    mask = torch.zeros_like(tensor)
    mask.scatter_(-1, topk_idx, 1.0) #top-k positions set to 1
    return tensor * mask

def apply_random_mask(tensor, ratio=0.5):
    mask = (torch.rand_like(tensor) < ratio).float()
    return tensor * mask

def compute_masked_ppl(model_name="distilgpt2", ratio=0.5, mode="topk",
                       log_list=None, num_samples=200, max_length=256, batch_size=8):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer.model_max_length = 1024
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    text_list = [t for t in dataset["validation"]["text"] if isinstance(t, str) and t.strip() != ""]
    sample_list = text_list[:num_samples]

    encoded = tokenizer(
        sample_list,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    token_lengths = attention_mask.sum(dim=1).tolist()
    avg_tokens = sum(token_lengths) / len(token_lengths)
    print(f"[Dataset Stats] Number of samples = {len(sample_list)}")
    print(f"[Dataset Stats] Average tokens per sample = {avg_tokens:.2f}")

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    def hook_fn(module, input, output):
        if mode == "topk":
            return apply_topk_mask(output, ratio)
        elif mode == "random":
            return apply_random_mask(output, ratio)
        return output

    handle = model.transformer.h[0].mlp.c_fc.register_forward_hook(hook_fn)

    total_nll = 0.0      
    total_tokens = 0    
    total_correct = 0  
    total_valid = 0   

    with torch.no_grad():
        num_samples = input_ids.size(0)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_input_ids = input_ids[start:end]
            batch_attention_mask = attention_mask[start:end]

            labels = batch_input_ids.clone()
            labels[batch_attention_mask == 0] = -100  # ignore pad

            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=labels,
            )
            loss = outputs.loss  
            logits = outputs.logits

            
            valid_mask = labels != -100
            n_tokens = valid_mask.sum().item()
            total_nll += loss.item() * n_tokens
            total_tokens += n_tokens

            preds = torch.argmax(logits, dim=-1)
            correct = (preds[valid_mask] == labels[valid_mask]).sum().item()
            total_correct += correct
            total_valid += n_tokens

    handle.remove()

    
    avg_nll = total_nll / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(avg_nll)
    acc = total_correct / total_valid if total_valid > 0 else 0.0

    msg = f"{mode.upper()}-{ratio:.2f} => PPL: {ppl:.2f}, ACC: {acc:.4f}"
    print(msg)

    if log_list is not None:
        log_list.append({
            "mode": mode,
            "ratio": ratio,
            "ppl": ppl,
            "acc": acc,
            "num_samples": num_samples,
            "avg_tokens": avg_tokens,
            "max_length": max_length,
        })


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(50)

    results = []
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

    for ratio in ratios:
        compute_masked_ppl(ratio=ratio, mode="topk", log_list=results)
        compute_masked_ppl(ratio=ratio, mode="random", log_list=results)


    with open("masked_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved to masked_results.json")


