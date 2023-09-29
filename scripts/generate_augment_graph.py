from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def main():
    file_dir = Path(__file__).resolve().parent
    colors = ["#003f5c", "#ffa600"]
    label = {
        "classification": "MolZip",
        "classification_vec": "MolZip-Vec",
        "regression": "MolZip",
        "regression_vec": "MolZip-Vec",
    }

    df = pd.read_csv(Path(file_dir, "../results/augment.csv"))

    for dataset in df["dataset"].unique():
        df_dataset = df[df["dataset"] == dataset]
        fig, ax1 = plt.subplots(1, 1, figsize=(4, 4))

        for i, task in enumerate(df_dataset["task"].unique()):
            df_task = df_dataset[df_dataset["task"] == task]
            ax1.plot(
                df_task["test_auroc"].to_list(),
                color=colors[i],
                label=f"{label[task]} (Test)",
            )
            ax1.plot(
                df_task["valid_auroc"].to_list(),
                linestyle="dashed",
                color=colors[i],
                label=f"{label[task]} (Valid)",
            )

            ax1.set_xlabel("Number of Augmentations")
            ax1.set_ylabel("AUROC" if "classification" in task else "RMSE")

        fig.legend(
            frameon=False,
            loc="upper center",
            ncols=2,
            bbox_to_anchor=(0.54, 1.10),
        )
        fig.tight_layout()
        fig.savefig(f"results/augment_{dataset}.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
