import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl


def plot_initial_attitudes(
        attitudes: np.ndarray,
        number_of_nodes: int
    ) -> None:
    """A convenience function for plotting initial attitudes."""
    f, ax = plt.subplots(1, 1, figsize=(6, 3))
    sns.scatterplot(
        x=np.arange(start=0, stop=number_of_nodes, step=1),
        y=attitudes,
        hue=attitudes,
        palette="PRGn",
        ax=ax
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.axhline(y=0, xmin=0, xmax=number_of_nodes, color="#C2CCDC")
    ax.yaxis.grid(True)
    ax.set_ylim(bottom=-1, top=1)
    sns.despine(trim=True, bottom=True)
    f.tight_layout()
    plt.show(f)


def plot_attitude_evolution(
        attitude_tracker: pl.DataFrame,
        tmax: int
    ) -> None:
    """Convenience function for plotting node attitudes over time"""
    f, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(y=0, xmin=0, xmax=tmax, color="#C2CCDC")
    sns.lineplot(
        ax=ax,
        data=attitude_tracker,
        x="time",
        y="attitude",
        hue="node",
        palette="flare",
        alpha=0.5
    )
    ax.xaxis.grid(True)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    sns.despine(bottom=True, trim=True)
    f.tight_layout()
    plt.show(f)
