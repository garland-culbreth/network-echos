"""Methods for plotting network model information."""
from typing import Literal, Self

import matplotlib.pyplot as plt
import seaborn as sns

from nim.core import NetworkModel


class NetworkPlot:
    """Object with basic plotting methods for model properties."""

    def __init__(
            self: Self,
            model: NetworkModel,
            theme: Literal["ticks", "whitegrid", "darkgrid"] = "ticks") -> Self:
        """Object to hold plots and plotting methods.

        Parameters
        ----------
        model : NetworkModel
            An instance of a nim.core.NetworkModel object.
        theme : str {'ticks', 'whitegrid'}, optional, default: 'ticks'
            Name of a Seaborn theme style. Used to set plot styling.

        Returns
        -------
        Self : Self@NetworkPlot
            An instance of the NetworkPlot object.

        """
        self.summary_table = model.summary_table
        if theme == "ticks":
            sns.set_theme(context="notebook", style="ticks")
        elif theme == "whitegrid":
            sns.set_theme(context="notebook", style="whitegrid")
        else:
            sns.set_theme(context="notebook", style="darkgrid")

    def attitude_and_connection_means(
            self:Self,
            fig_width: float = 7,
            fig_height: float = 3) -> Self:
        """Plot mean attitude over time.

        Parameters
        ----------
        fig_width : float, optional, default: 7
            Figure width, in inches.
        fig_height : float, optional, default: 3
            Figure height, in inches.

        Returns
        -------
        Self : Self@NetworkPlot
            An instance of the NetworkPlot object.

        """
        tmax = self.summary_table["time"].max()
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        ax0.hlines(y=0, xmin=0, xmax=tmax, color="#444444")
        sns.lineplot(
            data=self.summary_table,
            x="time",
            y="attitude_mean",
            ax=ax0)
        sns.lineplot(
            data=self.summary_table,
            x="time",
            y="connection_mean",
            ax=ax1)
        ax0.set_xscale("log")
        ax1.set_xscale("log")
        sns.despine(ax=ax0)
        sns.despine(ax=ax1)
        fig.tight_layout()
        self.means_over_time = fig
        return self
