import matplotlib.pyplot as plt
import json
from collections import defaultdict

def plot_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    results = defaultdict(dict)
    for item in data:
        ratio = float(item["ratio"])
        mode = item["mode"]
        ppl = item["ppl"]
        acc = item.get("acc", None)

        results[ratio][mode] = {"ppl": ppl, "acc": acc}


    ratios = sorted(results.keys())
    topk_ppl = [results[r]["topk"]["ppl"] for r in ratios if "topk" in results[r]]
    randomk_ppl = [results[r]["random"]["ppl"] for r in ratios if "random" in results[r]]

    topk_acc = [results[r]["topk"]["acc"] for r in ratios if "topk" in results[r]]
    randomk_acc = [results[r]["random"]["acc"] for r in ratios if "random" in results[r]]


    plt.figure(figsize=(8, 6))
    plt.plot(ratios, topk_ppl, marker="o", label="Top-K PPL")
    plt.plot(ratios, randomk_ppl, marker="o", label="Random-K PPL")
    plt.xlabel("Density (ratio of active neurons)")
    plt.ylabel("Perplexity")
    plt.title("DistilGPT-2 on WikiText-2: PPL vs Density")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("ppl_topk_vs_random.png")
    print("plot to ppl_topk_vs_random.png")

# 圖 2: Accuracy vs density
    plt.figure(figsize=(8, 6))
    plt.plot(ratios, topk_acc, marker="o", label="Top-K Accuracy")
    plt.plot(ratios, randomk_acc, marker="o", label="Random-K Accuracy")
    plt.xlabel("Density (ratio of active neurons)")
    plt.ylabel("Token Accuracy")
    plt.title("DistilGPT-2 on WikiText-2: Accuracy vs Density")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("acc_topk_vs_random.png")
    print("Saved to acc_topk_vs_random.png")

if __name__ == "__main__":
    plot_from_json("masked_results.json")
