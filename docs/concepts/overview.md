# Overview

Network echos treats the dynamics of a network self-organizing in response to information diffusion as the result of two dynamical network properties: connections and attitudes.

## Connections

Connections are represented by the adjacency matrix of the network as weighted edges. The probability of two nodes interacting at each time step is equal to the weight of the edge connecting them.

## Attitudes

Attitudes are represented as positions on the unit semicircle from $[-\frac{\pi}{2}, \frac{\pi}{2}$]. This maps the sin of each attitude to the interval $[-1, 1]$.

## Interaction and adaptation

Each time step, nodes are randomly interact with the weights of edges in the adjacency matrix acting as interaction probabilities. When two nodes itneract, each node's attitude is reinforced, positively or negatively, by an amount proportional to the weight of the edge connecting them.

## Control parameters

The model uses two parameters, $\alpha$ and $\beta$, to govern the reinfocement dynamics. The $\alpha$ parameter tunes the amount by which nodes' attitudes are reinforced by acting as an exponent on the adjacency matrix. The sign of this parameter determines whether interaction with opposing attitude nodes strengthens (positive) or weakens (negative) the nodes' attitudes.

The parameter $\beta$ governs the speed at which nodes change their attitudes. It is a constant which modulates the magnitude of attitude change. When $\beta$ is large, attitudes change rapidly, and as $\beta$ becomes smaller they change more slowly. This parameter is mostly useful to avoid needing to simulate a very large number of time steps to see an effect.
