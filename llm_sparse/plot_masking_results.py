import matplotlib.pyplot as plt
import json
from collections import defaultdict

def plot_from_json(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    # 資料整理結構: data_map[target][mode] = list of dicts
    # target: "first" or "all"
    # mode: "topk" or "random"
    results = defaultdict(lambda: defaultdict(list))
    
    for item in data:
        target = item.get("target", "first") # 相容舊格式，預設 first
        mode = item["mode"]
        results[target][mode].append(item)

    # 排序並提取數據的輔助函式
    def get_xy(target, mode, metric_key):
        # 取出該組列表
        items = results[target][mode]
        # 根據 ratio 排序
        items.sort(key=lambda x: float(x["ratio"]))
        
        ratios = [float(x["ratio"]) for x in items]
        values = [float(x[metric_key]) for x in items]
        return ratios, values

    # 設定繪圖樣式
    # 格式: (target, mode, label, color, linestyle)
    plot_configs = [
        ("all", "topk", "All Layers - TopK", "blue", "-"),      # 實線
        ("all", "random", "All Layers - Random", "orange", "-"), # 實線
        ("first", "topk", "Layer 0 - TopK", "blue", "--"),       # 虛線
        ("first", "random", "Layer 0 - Random", "orange", "--")  # 虛線
    ]

    # --- 圖 1: PPL vs Density ---
    plt.figure(figsize=(10, 6)) # 稍微加寬一點
    
    for target, mode, label, color, ls in plot_configs:
        if target in results and mode in results[target]:
            ratios, ppls = get_xy(target, mode, "ppl")
            plt.plot(ratios, ppls, marker="o", label=label, color=color, linestyle=ls)

    plt.xlabel("Density (ratio of active neurons)")
    plt.ylabel("Perplexity (Lower is better)")
    plt.title("DistilGPT-2: PPL Comparison (First Layer vs All Layers)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # 因為 Random Mask 全層的 PPL 可能會爆高，建議開 Log Scale (看情況可註解掉)
    plt.yscale("log") 
    plt.tight_layout()
    plt.savefig("comparison_ppl.png")
    print("Saved to comparison_ppl.png (Log Scale)")

    # --- 圖 2: Accuracy vs Density ---
    plt.figure(figsize=(10, 6))
    
    for target, mode, label, color, ls in plot_configs:
        if target in results and mode in results[target]:
            ratios, accs = get_xy(target, mode, "acc")
            plt.plot(ratios, accs, marker="o", label=label, color=color, linestyle=ls)

    plt.xlabel("Density (ratio of active neurons)")
    plt.ylabel("Token Accuracy (Higher is better)")
    plt.title("DistilGPT-2: Accuracy Comparison (First Layer vs All Layers)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("comparison_acc.png")
    print("Saved to comparison_acc.png")

if __name__ == "__main__":
    # 注意：這裡讀取的檔名要跟上一步存的一樣
    plot_from_json("masked_results_comparison.json")