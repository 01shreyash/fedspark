from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curves(
    csv_dir: str = "results",
    output_dir: str = "docs/figures",
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_dir = Path(csv_dir)
    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found. Run experiments first.")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for csv_file in sorted(csv_files):
        df = pd.read_csv(csv_file)
        label = csv_file.stem
        ax1.plot(df["round"], df["auc_roc"], label=label)
        ax2.plot(df["round"], df["auc_pr"], label=label)
    ax1.set_xlabel("Round"); ax1.set_ylabel("AUC-ROC"); ax1.set_title("AUC-ROC vs Rounds"); ax1.legend()
    ax2.set_xlabel("Round"); ax2.set_ylabel("AUC-PR"); ax2.set_title("AUC-PR vs Rounds"); ax2.legend()
    plt.tight_layout()
    plt.savefig(out / "learning_curves.png", dpi=150)
    print(f"Saved {out / 'learning_curves.png'}")


def generate_results_table(csv_dir: str = "results", output_path: str = "RESULTS.md"):
    csv_dir = Path(csv_dir)
    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found.")
        return
    rows = []
    for csv_file in sorted(csv_files):
        df = pd.read_csv(csv_file)
        final = df.iloc[-1] if len(df) > 0 else {}
        rows.append({
            "experiment": csv_file.stem,
            "rounds": len(df),
            "final_auc_roc": f"{final.get('auc_roc', 0):.4f}",
            "final_auc_pr": f"{final.get('auc_pr', 0):.4f}",
        })
    md = "# Results\n\n"
    md += "| Experiment | Rounds | Final AUC-ROC | Final AUC-PR |\n"
    md += "|---|---|---|---|\n"
    for r in rows:
        md += f"| {r['experiment']} | {r['rounds']} | {r['final_auc_roc']} | {r['final_auc_pr']} |\n"
    with open(output_path, "w") as f:
        f.write(md)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    plot_learning_curves()
    generate_results_table()
