# Overview

Network echos treats the dynamics of a network self-organizing in response to information diffusion as the result of two network properties, connections and attitudes, which are interdependent and change over time according to a reinforecement mechanism.

## Connections and attitudes

Connections are represented by the adjacency matrix of the network as weighted edges with weights constrained to the interval $[0, 1]$. These edge weights are used as the probability of each pair of nodes interacting over time.

Attitudes are represented as positions on the unit semicircle from $[-\frac{\pi}{2}, \frac{\pi}{2}$]. This maps the sine of each attitude to the interval $[-1, 1]$.

## Interaction and adaptation

Each time step, nodes interact randomly with the weights of edges in the adjacency matrix acting as interaction probabilities. When two nodes interact, each node's attitude is reinforced, positively or negatively according to whether they agree or disagree, by an amount proportional to the magnitude of attitude difference between them and their existing edge weight. Simultaneously, the weight of the edge connecting the interacting nodes is also reinforced, again positively or negatively, by an amount proportional to the magnitude of attitude difference between them.

## Reinforcement dynamics

The model uses two parameters, $\alpha$ and $\beta$, to govern the reinforcement dynamics. The $\alpha$ parameter tunes the amount by which nodes' attitudes are reinforced by acting as an exponent on the adjacency matrix. The sign of this parameter determines whether interaction with opposing attitude nodes strengthens (positive) or weakens (negative) the nodes' attitudes.

The parameter $\beta$ governs the speed at which nodes change their attitudes. It is a constant which modulates the magnitude of attitude reinforcement. When $\beta$ is large, attitudes change rapidly, and as $\beta$ becomes smaller they change more slowly. The sign of this parameter governs whether the network tends to polarize or synchronize over time. If positive, as default, the reinforcemnt tends to polarize the network, if negative the reinforcement tends to synchronize the network.
