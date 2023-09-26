from pathlib import Path
import tmap as tm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from molzip import ZipKNNGraph
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import MolFromSmiles


def main():
    fig, ax1 = plt.subplots(1, 1)
    cmap = ListedColormap(["#003f5c", "#ffa600"])

    file_dir = Path(__file__).resolve().parent
    data_file = Path(file_dir, "../data/BBBP.csv")

    df = pd.read_csv(data_file)
    df["label"] = df["p_np"]

    # MolZip
    zg = ZipKNNGraph()

    edge_list = zg.fit_predict(df["smiles"].to_list(), 5)

    cfg = tm.LayoutConfiguration()
    cfg.node_size = 1
    cfg.mmm_repeats = 2
    cfg.sl_repeats = 2

    x, y, s, t, _ = tm.layout_from_edge_list(
        len(df), edge_list, create_mst=True, config=cfg
    )

    # Plot the edges
    for i in range(len(s)):
        ax1.plot(
            [x[s[i]], x[t[i]]],
            [y[s[i]], y[t[i]]],
            "k-",
            linewidth=0.5,
            alpha=0.5,
            zorder=1,
        )

    # Plot the vertices
    ax1.scatter(x, y, zorder=2, s=5, c=df["label"].to_list(), cmap=cmap)
    ax1.set_aspect("equal")
    ax1.axes.xaxis.set_ticklabels([])
    ax1.axes.yaxis.set_ticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    fig.tight_layout()
    fig.savefig("molzip.pdf", dpi=300, bbox_inches="tight")

    # ECFP
    fig, ax1 = plt.subplots(1, 1)
    fpgen = GetMorganGenerator(radius=3)

    X = []
    y = []

    for i, row in df.iterrows():
        mol = MolFromSmiles(row["smiles"])

        if mol:
            X.append(fpgen.GetFingerprintAsNumPy(mol))
            y.append(row["label"])

    te = tm.embed(
        np.array(X),
        layout_generator=tm.layout_generators.AnnoyLayoutGenerator(
            node_size=1, mmm_repeats=2, sl_repeats=2
        ),
    )

    tm.plot(
        te,
        show=False,
        line_kws={"linestyle": "--", "color": "gray"},
        scatter_kws={"s": 5, "c": y, "cmap": cmap},
        ax=ax1,
    )
    ax1.set_aspect("equal")

    ax1.axes.xaxis.set_ticklabels([])
    ax1.axes.yaxis.set_ticklabels([])

    ax1.set_xticks([])
    ax1.set_yticks([])
    fig.tight_layout()
    fig.savefig("ecfp.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
