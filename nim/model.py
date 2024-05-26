import numpy as np
import networkx as nx
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
    g_social : nx.Graph
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
    assert network_type in supported_types, f"Parameter network_type must be \
        one of {supported_types}"
    if network_type == "complete":
        g_social = nx.complete_graph(n=number_of_nodes)
    if network_type == "gnp_random":
        g_social = nx.gnp_random_graph(n=number_of_nodes, p=p)
    if network_type == "watts_strogatz":
        g_social = nx.watts_strogatz_graph(n=number_of_nodes, k=k, p=p)
    if network_type == "newman_watts_strogatz":
        g_social = nx.newman_watts_strogatz_graph(number_of_nodes, k=k, p=p)
    if network_type == "barabasi_albert":
        g_social = nx.barabasi_albert_graph(n=number_of_nodes, m=m)
    return g_social


def initialize_attitudes(
        number_of_nodes: int,
        distribution: str = "normal",
        loc: float = 0.0,
        scale: float = 1
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
    assert distribution in supported_distibutions, f"Parameter 'distribution' \
        must be one of {supported_distibutions}."
    if distribution == "normal":
        rng = np.random.default_rng()
        attitudes = rng.normal(loc=loc, scale=scale, size=number_of_nodes)
    attitudes = np.where(attitudes > np.pi/2, np.pi/2, attitudes)
    attitudes = np.where(attitudes < -np.pi/2, -np.pi/2, attitudes)
    assert np.isfinite(attitudes).all(), "attitudes has NaNs."
    return attitudes


def initialize_edges(
        g_social: nx.Graph,
        neighbor_weight: float = 1.0,
        non_neighbor_weight: float = 0.0
    ) -> np.ndarray:
    """Set the initial network edge weights

    Parameters
    ----------
    g_social : nx.Graph
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
    a_social : np.ndarray
        The network's adjacency matrix as a 2-dimensional
        numpy array.
    """
    a_social = nx.to_numpy_array(g_social)
    a_social = neighbor_weight * a_social
    a_social[a_social == 0] = non_neighbor_weight
    assert np.isfinite(a_social).all(), "a_social has NaNs."
    return a_social


def make_interactions(
        a_social: np.ndarray,
        number_of_nodes: int,
        reciprocate: bool = True
    ) -> np.ndarray:
    """Makes the random interactions at each time step"""
    assert np.isfinite(a_social).all(), "a_social has NaNs."
    rng = np.random.default_rng()
    a_interaction = rng.random(size=(number_of_nodes, number_of_nodes))
    a_interaction = np.where(a_interaction <= a_social, 1, 0)
    if reciprocate:
        a_interaction = np.maximum(a_interaction, a_interaction.transpose())
    return a_interaction


def update_edges(
        a_social: np.ndarray,
        a_interaction: np.ndarray,
        attitudes: np.ndarray,
        method: str = "type1"
    ) -> np.ndarray:
    """Update network edges based on interactions and attitudes

    Parameters
    ----------
    a_social : np.ndarray
        Adjacency matrix of the social network. Used to
        weight attitude reinforcement.
    a_interaction : np.ndarray
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
    a_social : np.ndarray
        Updated adjacency matrix of the social network.

    Raises
    ------
    AssertionError
        If `method` is not one of the supported_methods.
    AssertionError
        If `a_social` contains any NaN values.
    AssertionError
        If `a_interaction` contains any NaN values.
    AssertionError
        If `attitudes` contains any NaN values.
    """
    supported_methods = ["type1", "type2", "type3", "type4"]
    assert method in supported_methods, f"Parameter method must be one of \
        {supported_methods}"
    assert np.isfinite(a_social).all(), "a_social has NaNs."
    assert np.isfinite(a_interaction).all(), "a_interaction has NaNs."
    assert np.isfinite(attitudes).all(), "attitudes has NaNs."
    for i, attitude_i in enumerate(attitudes):
        for j, attitude_j in enumerate(attitudes):
            if i == j:
                continue
            if method == "type1":
                # Consensus network with relaxing disturbances
                d_a_social = (a_interaction[i][j]
                            * (np.abs(attitude_i) - np.abs(attitude_j))
                            * np.sin(attitude_i * attitude_j))
            if method == "type2":
                d_a_social = (a_interaction[i][j]
                              * (np.abs(attitude_i) - np.abs(attitude_j))
                              * np.sin(attitude_i + attitude_j))
            if method == "type3":
                d_a_social = (a_interaction[i][j]
                              * np.sqrt((attitude_i - attitude_j)**2)
                              * np.sin(attitude_i * attitude_j))
            if method == "type4":
                d_a_social = (a_interaction[i][j]
                              * (np.abs(attitude_i - attitude_j)))
            if method == "type5":
                # Polarized network
                d_a_social = (a_interaction[i][j]
                              * (np.abs(attitude_i) - np.abs(attitude_j))
                              * -np.sin(attitude_i * attitude_j))
            a_social[i][j] = a_social[i][j] + d_a_social
    a_social = np.where(a_social < 1e-20, 1e-20, a_social)
    a_social = np.where(a_social > 1, 1, a_social)
    return a_social


def update_attitudes(
        a_social: np.ndarray,
        a_interaction: np.ndarray,
        attitudes: np.ndarray,
        method: str = "type1"
    ) -> np.ndarray:
    """Update node attitudes based on interactions

    Parameters
    ----------
    a_social : np.ndarray
        Adjacency matrix of the social network. Used to
        weight attitude reinforcement.
    a_interaction : np.ndarray
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
        If `a_social` contains any NaN values.
    AssertionError
        If `a_interaction` contains any NaN values.
    AssertionError
        If `attitudes` contains any NaN values.

    Note
    ----
    Setting `reinf_method_edges` to type4 and
    `reinf_method_atittude` to type4 is extremely
    interesting. They appear to create a model that
    gravitates toward consensus but has periods of intense
    instability.
    """
    supported_methods = ["type1", "type2", "type3", "type4", "type5"]
    assert method in supported_methods, f"Parameter method must be one of \
        {supported_methods}"
    assert np.isfinite(a_social).all(), "a_social has NaNs."
    assert np.isfinite(a_interaction).all(), "a_interaction has NaNs."
    assert np.isfinite(attitudes).all(), "attitudes has NaNs."
    for i, attitude_i in enumerate(attitudes):
        for j, attitude_j in enumerate(attitudes):
            if i == j:
                continue
            if method == "type1":
                # Noisy consensus with relaxing disturbances
                d_attitude = ((a_interaction[i][j] * a_social[i][j])
                              * np.abs(attitude_i - attitude_j)
                              * -np.sin(attitude_i + attitude_j))
            if method == "type2":
                # Polarized network
                d_attitude = ((a_interaction[i][j] / a_social[i][j])
                              * np.abs(attitude_i - attitude_j)
                              * np.sin(attitude_i + attitude_j))
            if method == "type3":
                # Extremely noisy almost anti-consensus network
                d_attitude = ((a_interaction[i][j] * a_social[i][j])
                              * (np.abs(attitude_i - attitude_j))
                              * np.sin(attitude_i * attitude_j))
            if method == "type4":
                # Noisy consensus with ongoing disturbances
                d_attitude = ((a_interaction[i][j] * a_social[i][j])
                              * np.sqrt((attitude_i - attitude_j)**2)
                              * np.sin(attitude_i * attitude_j))
            if method == "type5":
                # Polarized network
                d_attitude = ((a_interaction[i][j] * a_social[i][j])
                              * np.sin(attitude_i))
            attitudes[i] = attitudes[i] + d_attitude
    attitudes = np.where(attitudes < -np.pi/2, -np.pi/2, attitudes)
    attitudes = np.where(attitudes > np.pi/2, np.pi/2, attitudes)
    return attitudes


def run_model(
        tmax: int,
        number_of_nodes: np.ndarray,
        a_social: np.ndarray,
        attitudes: np.ndarray,
        reciprocate_interactions: bool = True,
        reinf_method_edges: str = "type1",
        reinf_method_atittude: str = "type1"
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
        "edge_weight_mean": np.mean(a_social),
        "edge_weight_median": np.median(a_social),
        "edge_weight_sd": np.std(a_social),
        "attitude_mean": np.mean(attitudes),
        "attitude_median" : np.median(attitudes),
        "attitude_sd": np.std(attitudes)
    })
    # Run the simulation
    for t in tqdm(range(tmax)):
        a_interaction = make_interactions(
            a_social,
            number_of_nodes,
            reciprocate_interactions
        )
        a_social = update_edges(
            a_social,
            a_interaction,
            attitudes,
            reinf_method_edges
        )
        attitudes = update_attitudes(
            a_social,
            a_interaction,
            attitudes,
            reinf_method_atittude
        )
        # Add new rows to track nodes over time
        attitudes_t = pl.DataFrame({
            "time": np.repeat(t, repeats=number_of_nodes),
            "node": np.arange(number_of_nodes),
            "attitude": attitudes
        })
        sim_summary_t = pl.DataFrame({
            "time": t,
            "edge_weight_mean": np.mean(a_social),
            "edge_weight_median": np.median(a_social),
            "edge_weight_sd": np.std(a_social),
            "attitude_mean": np.mean(attitudes),
            "attitude_median" : np.median(attitudes),
            "attitude_sd": np.std(attitudes)
        })
        attitude_tracker = pl.concat([attitude_tracker, attitudes_t])
        sim_summary = pl.concat([sim_summary, sim_summary_t])
    return [attitude_tracker, sim_summary]
