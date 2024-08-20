"""Core functionality for the network infodemic model.

Defines the core class and methods for constructing and running an
instance of the model.
"""
from typing import Literal, Self

import networkx as nx
import numpy as np
import polars as pl
from tqdm import tqdm


class NetworkModel:
    """Object containing network model parameters and methods."""

    def __init__(
            self: Self,
            number_of_nodes: int,
            interaction_type: str,
            alpha: float = -1,
            beta: float = 1e-3) -> Self:
        """Object storing network model data and methods.

        Parameters
        ----------
        number_of_nodes : int
            Number of nodes to add to the network model.
        interaction_type : str {"symmetric", "asymmetric"}
            Type of interaction between nodes. If "symmetric"
            interactions will be reciprocated. If "asymmetric",
            interactions may be one-directional.
        alpha : float
            Exponent of the adjacency matrix during attitude
            reinforcement.
        beta : float
            Factor governing the rate of attitude change. Smaller
            values slow attitude change.

        Returns
        -------
        self : Self@NetworkModel
            An instance of the NetworkModel object.

        """
        allowed_types = ["symmetric", "asymmetric"]
        if interaction_type not in allowed_types:
            msg = f"interaction_type must be one of: {allowed_types}."
            raise ValueError(msg)
        if number_of_nodes < 1:
            msg = "number_of_nodes must be greater than one."
            raise ValueError(msg)
        self.number_of_nodes = number_of_nodes
        self.interaction_type = interaction_type
        self.adjacency_exponent = alpha
        self.attitude_change_speed = beta
        self.network_type = None
        self.k_neighbors = None
        self.p_edge = None
        self.m_new_node_edges = None
        self.neighbor_weight = None
        self.non_neighbor_weight = None
        self.social_network = None
        self.connections = None
        self.attitudes = None
        self.attitude_distribution = None
        self.attitude_distribution_loc = None
        self.attitude_distribution_scale = None
        self.attitude_distribution_low = None
        self.attitude_distribution_high = None
        self.attitude_distribution_mu = None
        self.attitude_distribution_kappa = None
        self.interactions = None
        self.attitude_diffs = None
        self.summary_table = None
        self.attitude_tracker = None

    def create_network(
            self: Self,
            network_type: Literal[
                "complete",
                "erdos_renyi",
                "watts_strogatz",
                "newman_watts_strogatz",
                "barabasi_albert"],
            p: float = 0.1,
            k: int = 2,
            m: float = 1.0) -> Self:
        """Create initial social network.

        Parameters
        ----------
        network_type: str
            Type of network to create.
        p : float, optional, default: 0.1
            Required if network_type is one of {'erdos_renyi',
            'watts_strogatz', 'newman_watts_strogatz'}. Probability for
            edge creation.
        k : int, optional, default: 2
            Required if network_type is one of {'watts_strogatz',
            'newman_watts_strogatz'}. Each node is joined with its `k`
            nearest neighbors in a ring topology.
        m : float, optional, default: 1.0
            Required if network_type is 'barabasi_albert'. Number
            of edges to attach from a new node to existing nodes.

        Returns
        -------
        self : Self@NetworkModel
            An instance of the NetworkModel object.

        """
        if network_type == "complete":
            g_initial = nx.complete_graph(n=self.number_of_nodes)
        if network_type == "erdos_renyi":
            g_initial = nx.erdos_renyi_graph(n=self.number_of_nodes, p=p)
            self.p_edge = p
        if network_type == "watts_strogatz":
            g_initial = nx.watts_strogatz_graph(
                n=self.number_of_nodes, k=k, p=p)
            self.k_neighbors = k
            self.p_edge = p
        if network_type == "newman_watts_strogatz":
            g_initial = nx.newman_watts_strogatz_graph(
                n=self.number_of_nodes, k=k, p=p)
            self.k_neighbors = k
            self.p_edge = p
        if network_type == "barabasi_albert":
            g_initial = nx.barabasi_albert_graph(n=self.number_of_nodes, m=m)
            self.m_new_node_edges = m
        self.social_network = g_initial
        self.network_type = network_type
        return self

    def initialize_connections(
            self: Self,
            neighbor_weight: float = 1.0,
            non_neighbor_weight: float = 0.0) -> Self:
        """Set initial network edge weights.

        Parameters
        ----------
        neighbor_weight : float
            The weight to assign to edges between nodes which have an
            existing edge in the intial network.
        non_neighbor_weight : float
            The weight to assign to edges between nodes which don't
            have an existing edge in the intial network. If non-zero
            the network will technically become a complete network.

        Returns
        -------
        self : Self@NetworkModel
            An instance of the NetworkModel object.

        """
        adjacency_mat = nx.to_numpy_array(self.social_network)
        adjacency_mat = neighbor_weight * adjacency_mat
        adjacency_mat[adjacency_mat == 0] = non_neighbor_weight
        self.connections = adjacency_mat
        self.neighbor_weight = neighbor_weight
        self.non_neighbor_weight = non_neighbor_weight
        return self

    def initialize_attitudes(
            self: Self,
            distribution: Literal[
                "normal",
                "uniform",
                "laplace",
                "vonmises"] = "normal",
            a: float = 0.0,
            b: float = 0.3) -> Self:
        """Set initial node attitudes.

        Parameters
        ----------
        distribution : str {'normal', 'uniform', 'laplace', 'vonmises'}, default: 'normal'
            Type of probability distribution to sample attitudes from.
        a : float, default: 0.0
            First parameter for `distribution`. If `distribution` is
            'normal' or 'laplace' this is the `loc` parameter. If
            `distribution` is 'uniform', this is the lower bound. If
            `distribution` is 'vonmises' this is the `mu` parameter.
        b : float, default: 0.3
            Second parameter for `distribution`. If `distribution` is
            'normal' or 'laplace' this is the `scale` parameter. If
            `distribution` is 'uniform', this is the upper bound. If
            `distribution` is 'vonmises' this is the `kappa` parameter.

        Returns
        -------
        self : Self@NetworkModel
            An instance of the NetworkModel object.

        """  # noqa: E501
        rng = np.random.default_rng()
        if distribution == "normal":
            attitudes = rng.normal(
                loc=a, scale=b, size=(1, self.number_of_nodes))
            self.attitude_distribution_loc = a
            self.attitude_distribution_scale = b
        if distribution == "laplace":
            attitudes = rng.laplace(
                loc=a, scale=b, size=(1, self.number_of_nodes))
            self.attitude_distribution_loc = a
            self.attitude_distribution_scale = b
        if distribution == "vonmises":
            attitudes = rng.vonmises(
                mu=a, kappa=b, size=(1, self.number_of_nodes))
            self.attitude_distribution_mu = a
            self.attitude_distribution_kappa = b
        if distribution == "uniform":
            attitudes = rng.uniform(
                low=a, high=b, size=(1, self.number_of_nodes))
            self.attitude_distribution_low = a
            self.attitude_distribution_high = b
        attitudes = np.clip(attitudes, a_min=-np.pi/2, a_max=np.pi/2)
        self.attitudes = attitudes
        self.attitude_distribution = distribution
        return self

    def make_symmetric_interactions(self: Self) -> Self:
        """Construct a random symmetric adjacency matrix."""
        rng = np.random.default_rng()
        rand_mat = rng.random(
            size=(self.number_of_nodes, self.number_of_nodes))
        interactions = np.where(rand_mat <= self.connections, 1, 0)
        interactions = np.maximum(interactions, interactions.transpose())
        self.interactions = interactions
        return self

    def make_asymmetric_interactions(self: Self) -> Self:
        """Construct a random asymmetric adjacency matrix."""
        rng = np.random.default_rng()
        rand_mat = rng.random(
            size=(self.number_of_nodes, self.number_of_nodes))
        interactions = np.where(rand_mat <= self.connections, 1, 0)
        self.interactions = interactions
        return self

    def make_interactions(self: Self) -> Self:
        """Construct interaction matrix."""
        if self.interaction_type == "symmetric":
            self.make_symmetric_interactions()
        elif self.interaction_type == "asymmetric":
            self.make_asymmetric_interactions()
        else:
            msg = f"""self.interaction_type must be one of ["symmetric",
            "asymmetric"], got: {self.interaction_type}"""
            raise ValueError(msg)
        return self

    def compute_attitude_difference_matrix(self: Self) -> Self:
        """Construct matrix of differences between node attitudes."""
        self.attitude_diffs = np.subtract(
            self.attitudes,
            self.attitudes.transpose())
        return self

    def update_connections(self: Self) -> Self:
        """Calculate change in connection strength."""
        d_connections = np.multiply(
            self.interactions,
            np.sin(self.attitude_diffs))
        self.connections = self.connections + d_connections
        self.connections = np.clip(self.connections, a_min=0, a_max=1)
        return self

    def update_attitudes(self: Self) -> Self:
        """Calculate change in attitude."""
        d_attitudes = np.sum(
            np.multiply(
                np.multiply(
                    self.attitudes**self.adjacency_exponent, self.interactions),
                np.sin(self.attitude_diffs)),
            axis=1)
        self.attitudes = (self.attitudes
                          + (self.attitude_change_speed * d_attitudes))
        self.attitudes = np.clip(self.attitudes, a_min=-np.pi/2, a_max=np.pi/2)
        return self

    def initialize_summary_table(self: Self) -> Self:
        """Initialize summary table."""
        self.summary_table = pl.DataFrame(schema={
            "time": int,
            "attitude_mean": float,
            "attitude_sd": float,
            "connection_mean": float,
            "connection_sd": float})
        return self

    def update_summary_table(self: Self, time: int) -> Self:
        """Construct summary of the network at one time."""
        step_summary_table = pl.DataFrame({
            "time": time,
            "attitude_mean": np.mean(np.sin(self.attitudes)),
            "attitude_sd": np.std(np.sin(self.attitudes)),
            "connection_mean": np.mean(self.connections),
            "connection_sd": np.std(self.connections)})
        self.summary_table = pl.concat(
            [self.summary_table, step_summary_table])
        return self

    def initialize_attitude_tracker(self: Self) -> Self:
        """Initialize attitude tracking table."""
        self.attitude_tracker = pl.DataFrame(schema={
            "time": int,
            "node": int,
            "attitude": float})
        return self

    def update_attitude_tracker(self: Self, time: int) -> Self:
        """Construct attitude tracking table for one time."""
        attitudes_flat = self.attitudes.flatten()
        step_attitude_tracker = pl.DataFrame({
            "time": np.full(
                shape=attitudes_flat.shape, fill_value=time, dtype="int64"),
            "node": np.arange(
                self.number_of_nodes, dtype="int64").reshape(
                    attitudes_flat.shape),
            "attitude": attitudes_flat})
        self.attitude_tracker = pl.concat(
            [self.attitude_tracker, step_attitude_tracker])
        return self

    def run_simulation(self: Self, tmax: int) -> Self:
        """Simulate interaction and attitude reinforcement over time.

        Parameters
        ----------
        tmax : int
            Number of time steps to iterate over.

        Returns
        -------
        self : Self@NetworkModel
            An instance of the NetworkModel object.

        """
        if tmax < 1:
            msg = "tmax must be greater than 1"
            raise ValueError(msg)
        self.initialize_summary_table()
        self.initialize_attitude_tracker()
        for t in tqdm(range(tmax)):
            self.compute_attitude_difference_matrix()
            self.make_interactions()
            self.update_connections()
            self.update_attitudes()
            self.update_summary_table(time=t)
            self.update_attitude_tracker(time=t)
