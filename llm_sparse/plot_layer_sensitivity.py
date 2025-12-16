import matplotlib.pyplot as plt
import json
from collections import defaultdict

def plot_layer_sensitivity(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found. Make sure you ran the experiment first!")
        return

    layer_data = defaultdict(dict)
    experiment_ratio = 0.5 
    baseline_ppl = None

    for item in data:
        mode = item["mode"]
        ratio = float(item["ratio"])
        ppl = item["ppl"]

        target_layers = item.get("target_layers", [])
        
        if not target_layers:
            continue
            
        layer_idx = target_layers[0]

        #Baseline(Ratio=1.0)
        if ratio == 1.0:
            baseline_ppl = ppl
            continue

        experiment_ratio = ratio
        layer_data[layer_idx][mode] = ppl

    layers = sorted(layer_data.keys())
    topk_ppls = []
    random_ppls = []

    for layer in layers:
        topk_ppls.append(layer_data[layer].get("topk", None))
        random_ppls.append(layer_data[layer].get("random", None))

    plt.figure(figsize=(10, 6))
    
    #Top-K
    plt.plot(layers, topk_ppls, marker='o', label=f'Top-K (Ratio={experiment_ratio})', linewidth=2, color='#1f77b4')
    #Random
    plt.plot(layers, random_ppls, marker='x', linestyle='--', label=f'Random (Ratio={experiment_ratio})', linewidth=2, color='#ff7f0e')
    
    #畫Baseline
    if baseline_ppl:
        plt.axhline(y=baseline_ppl, color='gray', linestyle=':', label='Baseline (No Masking)', linewidth=1.5)

    plt.title(f'Layer-wise Sensitivity Analysis (DistilGPT-2)\nLower PPL is Better', fontsize=14)
    plt.xlabel('Layer Index (0 = First Layer, 5 = Last Layer)', fontsize=12)
    plt.ylabel('Perplexity (PPL)', fontsize=12)
    plt.xticks(layers)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    # 存檔
    output_filename = "layer_sensitivity.png"
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    plot_layer_sensitivity("layer_sensitivity_results.json")