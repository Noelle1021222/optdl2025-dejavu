import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import json
import random
import numpy as np
from tqdm import tqdm

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# --- 1. Masking 輔助函式 ---

def apply_topk_mask(tensor, ratio=0.5): 
    # ratio: keep ratio
    k = max(1, int(tensor.size(-1) * ratio))
    score = tensor.abs()
    topk_idx = torch.topk(score, k, dim=-1).indices 
    mask = torch.zeros_like(tensor)
    mask.scatter_(-1, topk_idx, 1.0) 
    return tensor * mask

def apply_random_mask(tensor, ratio=0.5):
    # ratio: keep ratio
    mask = (torch.rand_like(tensor) < ratio).float()
    return tensor * mask

def prepare_environment(model_name="distilgpt2", num_samples=200, max_length=256):
    print("Loading model and dataset... (This happens only once)")
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
    
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    print(f"[Ready] Device: {device}, Samples: {len(sample_list)}")
    return model, input_ids, attention_mask, device

def compute_masked_ppl(model, input_ids, attention_mask, 
                       ratio=0.5, mode="topk", target_layers=None, 
                       log_list=None, batch_size=8):

    if target_layers is None:
        target_layers = [0]
    if isinstance(target_layers, int):
        target_layers = [target_layers]

    def hook_fn(module, input, output):
        if mode == "topk":
            return apply_topk_mask(output, ratio)
        elif mode == "random":
            return apply_random_mask(output, ratio)
        return output

    handles = []
    for layer_idx in target_layers:
        if 0 <= layer_idx < len(model.transformer.h):
            #activation (GELU 後)
            h = model.transformer.h[layer_idx].mlp.act.register_forward_hook(hook_fn)
            handles.append(h)
        else:
            print(f"Warning: Layer {layer_idx} does not exist.")

    total_nll = 0.0      
    total_tokens = 0    
    total_correct = 0   
    total_valid = 0   

    # 開始評估
    with torch.no_grad():
        num_samples = input_ids.size(0)
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_input_ids = input_ids[start:end]
            batch_attention_mask = attention_mask[start:end]

            labels = batch_input_ids.clone()
            labels[batch_attention_mask == 0] = -100 

            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=labels,
            )
            loss = outputs.loss  
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 2. Valid Mask
            valid_mask = shift_labels != -100
            n_tokens = valid_mask.sum().item()
            
            total_nll += loss.item() * n_tokens
            total_tokens += n_tokens

            preds = torch.argmax(shift_logits, dim=-1)
            correct = (preds[valid_mask] == shift_labels[valid_mask]).sum().item()
            total_correct += correct
            total_valid += n_tokens

    # 移除 Hooks
    for h in handles:
        h.remove()

    avg_nll = total_nll / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(avg_nll)
    acc = total_correct / total_valid if total_valid > 0 else 0.0

    msg = f"Layer {target_layers} | {mode.upper()}-{ratio:.2f} => PPL: {ppl:.2f}, ACC: {acc:.4f}"
    print(msg)

    if log_list is not None:
        log_list.append({
            "mode": mode,
            "ratio": ratio,
            "target_layers": target_layers, 
            "ppl": ppl,
            "acc": acc,
            "num_samples": num_samples
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
    model, input_ids, attention_mask, device = prepare_environment()
    
    results = []
    test_ratio = 0.5   
    num_layers = 6     
    
    print("\n=== Step 1: Baseline (No Masking) ===")
    # Ratio 1.0 = No Masking
    compute_masked_ppl(model, input_ids, attention_mask, ratio=1.0, mode="topk", target_layers=[0], log_list=results)

    print(f"\n=== Step 2: Layer-wise Sensitivity Scan (Ratio={test_ratio}) ===")
    for layer_idx in range(num_layers):
        print(f"\n--- Testing Layer {layer_idx} ---")
        
        # 1. Top-K
        compute_masked_ppl(model, input_ids, attention_mask, 
                           ratio=test_ratio, mode="topk", target_layers=[layer_idx], log_list=results)
        
        # 2. Random-K
        compute_masked_ppl(model, input_ids, attention_mask, 
                           ratio=test_ratio, mode="random", target_layers=[layer_idx], log_list=results)

    # 存檔
    output_filename = "layer_sensitivity_results.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_filename}")