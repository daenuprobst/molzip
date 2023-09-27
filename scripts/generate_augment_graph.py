from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def main():
    file_dir = Path(__file__).resolve().parent

    df = pd.read_csv(Path(file_dir, "../results/augment.csv"))
    fig, ax1 = plt.subplots(1, 1)

    fig.tight_layout()
    fig.savefig("ecfp.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
