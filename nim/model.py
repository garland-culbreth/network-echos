import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from tqdm import tqdm


def create_network(
        network_type: str,
        number_of_nodes: int,
        k: int = 2,
        p: float = 0.1,
        m: float = 1.0
    ) -> nx.Graph:
    """Create a network of nodes and edges

    Parameters
    ----------
    network_type : str
        The type of network to be created. Must be a
        network class supported by networkx.
    number_of_nodes : int
        The number of nodes to use for the network.
    k : int, {optional, default: 2}
        The number of neighbors each node should have an
        edge to. For watts-strogatz and related.
    p : float, {optional, default: 0.1}
        If network_type=="gnp": The probablity of nodes
        having an edge. If network_type=="watts_strogatz":
        The probability that an edge will be re-wired.
    m : float, {optional, default: 1.0}
        If network_type=="barabasi_albert": The index of
        the power-law characterizing the degree
        distribution.
    
    Returns
    -------
    G_social : nx.Graph
        A networkx graph object of the network.

    Raises
    ------
    AssertionError
        If "network_type" is not one of the supported types.
    """
    supported_types = [
        "complete",
        "gnp_random",
        "watts_strogatz",
        "newman_watts_strogatz",
        "barabasi_albert"
    ]
    assert network_type in supported_types, f"Parameter network_type must be one of {supported_types}"
    if network_type == "complete":
        G_social = nx.complete_graph(n=number_of_nodes)
    if network_type == "gnp_random":
        G_social = nx.gnp_random_graph(n=number_of_nodes, p=p)
    if network_type == "watts_strogatz":
        G_social = nx.watts_strogatz_graph(n=number_of_nodes, k=k, p=p)
    if network_type == "newman_watts_strogatz":
        G_social = nx.newman_watts_strogatz_graph(number_of_nodes, k=k, p=p)
    if network_type == "barabasi_albert":
        G_social = nx.barabasi_albert_graph(n=number_of_nodes, m=m)
    return G_social


def initialize_attitudes(
        number_of_nodes: int,
        distribution: str = "normal",
        loc: float = 0.0,
        scale: float = 0.1
    ) -> np.ndarray:
    """Generate and set the initial attitudes.

    Parameters
    ----------
    number_of_nodes : int
        The number of nodes in the network.
    distribution : str, {optional, default: "normal"}
        The name of a probability density from which to
        sample the attitudes. Must be a distrubition
        supported by numpy.random.
    loc : float, {optional, default: 0.0}
        The location parameter for the probability distribution.
    scale : float, {optional, default: 0.1}
        The scale parameter for the probability distribution.

    Returns
    -------
    attitudes : np.ndarray
        A numpy array of the attitudes help by the nodes.
    
    Raises
    ------
    AssertionError
        If "distribution" is not one of the supported
        distributons.
    """
    supported_distibutions = ['normal']
    assert distribution in supported_distibutions, f"Parameter 'distribution' must be one of {supported_distibutions}."
    if distribution == "normal":
        rng = np.random.default_rng()
        attitudes = rng.normal(loc=loc, scale=scale, size=number_of_nodes)
    # attitudes[attitudes > 1.0] = 1.0
    # attitudes[attitudes < -1.0] = -1.0
    attitudes = np.where(attitudes > 1.0, 1.0, attitudes)
    attitudes = np.where(attitudes < -1.0, -1.0, attitudes)
    assert np.isfinite(attitudes).all(), "attitudes has NaNs."
    return attitudes


def initialize_edges(
        G_social: nx.Graph,
        neighbor_weight: float = 1.0,
        non_neighbor_weight: float = 0.0
    ) -> np.ndarray:
    """Set the initial network edge weights

    Parameters
    ----------
    G_social : nx.Graph
        The initial social network object whose edge are to
        be modified.
    neighbor_weight : float {optional, default: 1.0}
        The weight to give created edges.
    non_neighbor_weight : float {optional, default: 0.0}
        If non-zero, then edges with this weight will be
        created between all nodes which didn't previously
        have an edge.

    Returns
    -------
    A_social : np.ndarray
        The network's adjacency matrix as a 2-dimensional
        numpy array.
    """
    A_social = nx.to_numpy_array(G_social)
    A_social = neighbor_weight * A_social
    A_social[A_social == 0] = non_neighbor_weight
    assert np.isfinite(A_social).all(), "A_social has NaNs."
    return A_social


def make_interactions(
        A_social: np.ndarray,
        number_of_nodes: int,
        reciprocate: bool = True
    ) -> np.ndarray:
    assert np.isfinite(A_social).all(), "A_social has NaNs."
    rng = np.random.default_rng()
    A_interaction = rng.random(size=(number_of_nodes, number_of_nodes))
    A_interaction = np.where(A_interaction <= A_social, 1, 0)
    if reciprocate == True:
        A_interaction = np.maximum(A_interaction, A_interaction.transpose())
    return A_interaction


def update_edges(
        A_social: np.ndarray,
        A_interaction: np.ndarray,
        attitudes: np.ndarray,
        method: str = "type1"
    ) -> np.ndarray:
    """Update network edges based on interactions and attitudes

    Parameters
    ----------
    A_social : np.ndarray
        Adjacency matrix of the social network. Used to
        weight attitude reinforcement.
    A_interaction : np.ndarray
        Interaction matrix. Should contain only 1s and
        0s, indicating which nodes interacted at a given
        time step.
    attitudes : np.ndarray
        Array containing the attitudes to be updated.
    method : str {optional, detault: "type1"}
        Which reinforcement method to use. Details for the
        reinforcement types will be given in documentation
        (WIP).

    Returns
    -------
    A_social : np.ndarray
        Updated adjacency matrix of the social network.

    Raises
    ------
    AssertionError
        If `method` is not one of the supported_methods.
    AssertionError
        If `A_social` contains any NaN values.
    AssertionError
        If `A_interaction` contains any NaN values.
    AssertionError
        If `attitudes` contains any NaN values.
    """
    supported_methods = ["type1", "type2", "type3"]
    assert method in supported_methods, f"Parameter method must be one of {supported_methods}"
    assert np.isfinite(A_social).all(), "A_social has NaNs."
    assert np.isfinite(A_interaction).all(), "A_interaction has NaNs."
    assert np.isfinite(attitudes).all(), "attitudes has NaNs."
    for i in range(len(attitudes)):
        for j in range(len(attitudes)):
            if i == j:
                next
            if method == "type1":
                d_A_social = (A_interaction[i][j]
                              * (np.abs(attitudes[i]) - np.abs(attitudes[j]))
                              * np.sign(attitudes[i] * attitudes[j]))
            if method == "type2":
                d_A_social = (A_interaction[i][j]
                              * (np.abs(attitudes[i]) - np.abs(attitudes[j]))
                              * np.sign(attitudes[i] + attitudes[j]))
            if method == "type3":
                d_A_social = (A_interaction[i][j]
                              * np.sqrt((attitudes[i] - attitudes[j])**2)
                              * np.sign(attitudes[i] * attitudes[j]))
            A_social[i][j] = A_social[i][j] + d_A_social
    A_social = np.where(A_social < 1e-20, 1e-20, A_social)
    A_social = np.where(A_social > 1, 1, A_social)
    return A_social


def update_attitudes(
        A_social: np.ndarray,
        A_interaction: np.ndarray,
        attitudes: np.ndarray,
        method: str = "type1"
    ) -> np.ndarray:
    """Update node attitudes based on interactions

    Parameters
    ----------
    A_social : np.ndarray
        Adjacency matrix of the social network. Used to
        weight attitude reinforcement.
    A_interaction : np.ndarray
        Interaction matrix. Should contain only 1s and
        0s, indicating which nodes interacted at a given
        time step.
    attitudes : np.ndarray
        Array containing the attitudes to be updated.
    method : str {optional, detault: "type1"}
        Which reinforcement method to use. Details for the
        reinforcement types will be given in documentation
        (WIP).

    Returns
    -------
    attitudes : np.ndarray
        Array containing the new attitudes.

    Raises
    ------
    AssertionError
        If `method` is not one of the supported_methods.
    AssertionError
        If `A_social` contains any NaN values.
    AssertionError
        If `A_interaction` contains any NaN values.
    AssertionError
        If `attitudes` contains any NaN values.

    Note
    ----
    Setting method to type1 or type3 is extremely
    interesting. They appear to create a model that
    gravitates toward consensus but has periods of intense
    instability.
    """
    supported_methods = ["type1", "type2", "type3", "type4"]
    assert method in supported_methods, f"Parameter method must be one of {supported_methods}"
    assert np.isfinite(A_social).all(), "A_social has NaNs."
    assert np.isfinite(A_interaction).all(), "A_interaction has NaNs."
    assert np.isfinite(attitudes).all(), "attitudes has NaNs."
    for i in range(len(attitudes)):
        for j in range(len(attitudes)):
            if i == j:
                next
            if method == "type1":
                d_attitude = ((A_interaction[i][j] * A_social[i][j])
                              * np.abs(attitudes[i] - attitudes[j])
                              * np.sign(attitudes[i] + attitudes[j]))
            if method == "type2":
                d_attitude = ((A_interaction[i][j] / A_social[i][j])
                              * np.abs(attitudes[i] - attitudes[j])
                              * np.sign(attitudes[i] + attitudes[j]))
            if method == "type3":
                d_attitude = ((A_interaction[i][j] * A_social[i][j])
                              * (np.abs(attitudes[i] - attitudes[j]))
                              * np.sign(attitudes[i] * attitudes[j]))
            if method == "type4":
                d_attitude = ((A_interaction[i][j] * A_social[i][j])
                              * np.sqrt((attitudes[i] - attitudes[j])**2)
                              * np.sign(attitudes[i] * attitudes[j]))
            attitudes[i] = attitudes[i] - d_attitude
    attitudes = np.where(attitudes < -1, -1, attitudes)
    attitudes = np.where(attitudes > 1, 1, attitudes)
    return attitudes


def run_model(
        tmax: int,
        number_of_nodes: np.ndarray,
        A_social: np.ndarray,
        attitudes: np.ndarray,
        reciprocate_interactions: bool = True
    ) -> list[pl.DataFrame]:
    """Runs the model"""
    # Set up dataframe to track the attitudes over time
    attitude_tracker = pl.DataFrame({
        "time": np.repeat(0, repeats=number_of_nodes),
        "node": np.arange(number_of_nodes),
        "attitude": attitudes
    })
    sim_summary = pl.DataFrame({
        "time": 0,
        "edge_weight_mean": np.mean(A_social),
        "edge_weight_median": np.median(A_social),
        "edge_weight_sd": np.std(A_social),
        "attitude_mean": np.mean(attitudes),
        "attitude_median" : np.median(attitudes),
        "attitude_sd": np.std(attitudes)
    })
    # Run the simulation
    for t in tqdm(range(tmax)):
        A_interaction = make_interactions(
            A_social,
            number_of_nodes,
            reciprocate_interactions
        )
        A_social = update_edges(
            A_social,
            A_interaction,
            attitudes,
            method="type1"
        )
        attitudes = update_attitudes(
            A_social,
            A_interaction,
            attitudes,
            method="type1"
        )
        # Add new rows to track nodes over time
        attitudes_t = pl.DataFrame({
            "time": np.repeat(t, repeats=number_of_nodes),
            "node": np.arange(number_of_nodes),
            "attitude": attitudes
        })
        sim_summary_t = pl.DataFrame({
            "time": t,
            "edge_weight_mean": np.mean(A_social),
            "edge_weight_median": np.median(A_social),
            "edge_weight_sd": np.std(A_social),
            "attitude_mean": np.mean(attitudes),
            "attitude_median" : np.median(attitudes),
            "attitude_sd": np.std(attitudes)
        })
        attitude_tracker = pl.concat([attitude_tracker, attitudes_t])
        sim_summary = pl.concat([sim_summary, sim_summary_t])
    return [attitude_tracker, sim_summary]
