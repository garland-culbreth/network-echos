"""Tests for the core module."""
from typing import Self

import networkx as nx
import numpy as np
import polars as pl

import nim.core


class TestCore:
    """Class with methods for testing core functionality."""

    def test_create_network(self: Self) -> Self:
        """Test that basis social network constructs OK."""
        nimodel = nim.core.NetworkModel(
            number_of_nodes=10,
            interaction_type="symmetric")
        nimodel.create_network(network_type="gnp_random", p=0.1)
        assert hasattr(nimodel, "social_network"), \
            "Social network is missing."
        assert isinstance(nimodel.social_network, nx.Graph), \
            "Social network is not a networkx graph object."

    def test_initialize_connections(self: Self) -> Self:
        """Test that adjacency matrix attribute constructs OK."""
        nimodel = nim.core.NetworkModel(
            number_of_nodes=10,
            interaction_type="symmetric")
        nimodel.create_network(network_type="gnp_random", p=0.1)
        nimodel.initialize_connections()
        assert hasattr(nimodel, "connections"), \
            "Adjacency matrix is missing."
        assert isinstance(nimodel.connections, np.ndarray), \
            "Adjacency matrix is not a numpy ndarray."
        assert np.all(np.isfinite(nimodel.connections)), \
            "Adjacency matrix contains non-finite values."
        assert np.all(nimodel.connections >= 0), \
            "Adjacency matrix contains values less than zero."
        assert np.all(nimodel.connections <= 1), \
            "Adjacency matrix contains values greater than one."

    def test_initialize_attitudes(self: Self) -> Self:
        """Test that attitudes vector constructs OK."""
        nimodel = nim.core.NetworkModel(
            number_of_nodes=10,
            interaction_type="symmetric")
        nimodel.initialize_attitudes()
        assert hasattr(nimodel, "attitudes"), \
            "Attitude vector is missing."
        assert isinstance(nimodel.attitudes, np.ndarray), \
            "Attitude vector is not a numpy ndarray."
        assert np.all(np.isfinite(nimodel.attitudes)), \
            "Attitude vector contains non-finite values."
        assert np.all(nimodel.attitudes >= -np.pi/2), \
            "Attitude vector contains values less than -pi/2."
        assert np.all(nimodel.attitudes <= np.pi/2), \
            "Attitude vector contains values greater than pi/2."

    def test_make_symmetric_interactions(self: Self) -> Self:
        """Test that symmetric interaction matrices construct OK."""
        nimodel = nim.core.NetworkModel(
            number_of_nodes=10,
            interaction_type="symmetric")
        nimodel.create_network(network_type="gnp_random", p=0.1)
        nimodel.initialize_connections()
        nimodel.make_symmetric_interactions()
        assert hasattr(nimodel, "interactions"), \
            "Interaction matrix is missing."
        assert isinstance(nimodel.interactions, np.ndarray), \
            "Interaction matrix is not a numpy ndarray."
        assert np.all(np.isfinite(nimodel.interactions)), \
            "Interaction matrix contains non-finite values."
        assert np.all(nimodel.interactions >= 0), \
            "Interaction matrix contains values less than zero."
        assert np.all(nimodel.interactions <= 1), \
            "Interaction matrix contains values greater than one."
        assert np.all(
            nimodel.interactions == nimodel.interactions.transpose()), \
                "Interaction matrix is not symmetric."

    def test_make_asymmetric_interactions(self: Self) -> Self:
        """Test that asymmetric interaction matrices construct OK."""
        nimodel = nim.core.NetworkModel(
            number_of_nodes=10,
            interaction_type="asymmetric")
        nimodel.create_network(network_type="gnp_random", p=0.1)
        nimodel.initialize_connections()
        nimodel.make_asymmetric_interactions()
        assert hasattr(nimodel, "interactions"), \
            "Interaction matrix is missing."
        assert isinstance(nimodel.interactions, np.ndarray), \
            "Interaction matrix is not a numpy ndarray."
        assert np.all(np.isfinite(nimodel.interactions)), \
            "Interaction matrix contains non-finite values."
        assert np.all(nimodel.interactions >= 0), \
            "Interaction matrix contains values less than zero."
        assert np.all(nimodel.interactions <= 1), \
            "Interaction matrix contains values greater than one."

    def test_compute_attitude_difference_matrix(self: Self) -> Self:
        """Test that attitude difference matrix constructs OK."""
        nimodel = nim.core.NetworkModel(
            number_of_nodes=10,
            interaction_type="symmetric")
        nimodel.initialize_attitudes()
        nimodel.compute_attitude_difference_matrix()
        assert hasattr(nimodel, "attitude_diffs"), \
            "Attitude difference matrix is missing."
        assert isinstance(nimodel.attitude_diffs, np.ndarray), \
            "Attitude difference matrix is not a numpy ndarray."
        assert nimodel.attitude_diffs.shape == (
            nimodel.number_of_nodes,
            nimodel.number_of_nodes), \
                "Attitude difference matrix is not square."
        assert np.all(np.isfinite(nimodel.attitude_diffs)), \
            "Attitude difference matrix contains non-finite values."
        assert np.all(nimodel.attitude_diffs <= np.pi), \
            "Attitude difference matrix contains values greater than pi."

    def test_update_connections(self: Self) -> Self:
        """Test that connections update OK."""
        nimodel = nim.core.NetworkModel(
            number_of_nodes=10,
            interaction_type="symmetric")
        nimodel.create_network(network_type="gnp_random", p=0.1)
        nimodel.initialize_connections()
        nimodel.initialize_attitudes()
        nimodel.compute_attitude_difference_matrix()
        nimodel.make_symmetric_interactions()
        nimodel.update_connections()
        assert hasattr(nimodel, "connections"), \
            "Adjacency matrix is missing."
        assert isinstance(nimodel.connections, np.ndarray), \
            "Adjacency matrix is not a numpy ndarray."
        assert np.all(np.isfinite(nimodel.connections)), \
            "Adjacency matrix contains non-finite values."
        assert np.all(nimodel.connections >= 0), \
            "Adjacency matrix contains values less than zero."
        assert np.all(nimodel.connections <= 1), \
            "Adjacency matrix contains values greater than one."

    def test_update_attitudes(self: Self) -> Self:
        """Test that attitudes update OK."""
        nimodel = nim.core.NetworkModel(
            number_of_nodes=10,
            interaction_type="symmetric")
        nimodel.create_network(network_type="gnp_random", p=0.1)
        nimodel.initialize_connections()
        nimodel.initialize_attitudes()
        nimodel.compute_attitude_difference_matrix()
        nimodel.make_symmetric_interactions()
        nimodel.update_attitudes()
        assert hasattr(nimodel, "attitudes"), \
            "Attitude vector is missing."
        assert isinstance(nimodel.attitudes, np.ndarray), \
            "Attitude vector is not a numpy ndarray."
        assert np.all(np.isfinite(nimodel.attitudes)), \
            "Attitude vector contains non-finite values."
        assert np.all(nimodel.attitudes >= -np.pi/2), \
            "Attitude vector contains values less than -pi/2."
        assert np.all(nimodel.attitudes <= np.pi/2), \
            "Attitude vector contains values greater than pi/2."

    def test_initialize_summary_table(self: Self) -> Self:
        """Test that summary table initialized OK."""
        nimodel = nim.core.NetworkModel(
            number_of_nodes=10,
            interaction_type="symmetric")
        nimodel.initialize_summary_table()
        required_columns = [
            "time",
            "attitude_mean",
            "attitude_sd",
            "connection_mean",
            "connection_sd"]
        assert hasattr(nimodel, "summary_table"), \
            "Simulation summary table is missing."
        assert isinstance(nimodel.summary_table, pl.DataFrame), \
            "Simulation summary table is not a polars dataframe."
        for col in required_columns:
            assert col in nimodel.summary_table.columns, \
                f"Simulation summary table is missing column {col}."
        for col in required_columns:
            assert nimodel.summary_table[col].dtype.is_numeric(), \
                f"Simulation summary table column {col} has incorrect type."

    def test_update_summary_table(self: Self) -> Self:
        """Test that summary table initialized OK."""
        nimodel = nim.core.NetworkModel(
            number_of_nodes=10,
            interaction_type="symmetric")
        nimodel.create_network(network_type="gnp_random", p=0.1)
        nimodel.initialize_summary_table()
        nimodel.initialize_connections()
        nimodel.initialize_attitudes()
        nimodel.update_summary_table(time=1)
        required_columns = [
            "time",
            "attitude_mean",
            "attitude_sd",
            "connection_mean",
            "connection_sd"]
        assert hasattr(nimodel, "summary_table"), \
            "Simulation summary table is missing."
        assert isinstance(nimodel.summary_table, pl.DataFrame), \
            "Simulation summary table is not a polars dataframe."
        for col in required_columns:
            assert col in nimodel.summary_table.columns, \
                f"Simulation summary table is missing column {col}."
        for col in required_columns:
            assert nimodel.summary_table[col].dtype.is_numeric(), \
                f"Simulation summary table column {col} has incorrect type."
        for col in required_columns:
            assert nimodel.summary_table[col].is_finite().all(), \
                f"""Simulation summary table column {col} contains non-finite
                values."""


if __name__ == "__main__":
    test_core = TestCore()
    test_core.test_create_network()
    test_core.test_initialize_connections()
    test_core.test_initialize_attitudes()
    test_core.test_make_symmetric_interactions()
    test_core.test_make_asymmetric_interactions()
    test_core.test_compute_attitude_difference_matrix()
    test_core.test_update_connections()
    test_core.test_update_attitudes()
    test_core.test_initialize_summary_table()
    test_core.test_update_summary_table()
