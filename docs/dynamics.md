# Network and reinforcement dynamics

## Notation

| Variable | Definition |
|---|---|
| $t$ | Time. |
| $N$ | Number of nodes in the network. |
| $\mathbf{A}$ | The $N \times N$ adjacency matrix of the social network. A discrete function of time. Entries are the weights of the edges connecting the $i$-th and $j$-th nodes. |
| $\mathbf{T}$ | An $N \times N$ interaction matrix. Entries are $1$ or $0$ denoting whether nodes $i$ and $j$ interact at time $t$. |
| $\mathbf{M}$ | An $N \times N$ matrix of uniform random numbers between $0$ and $1$. Used to determine which nodes interact at time $t$. |
| $\mathbf{\theta}$ | A length $N$ column vector of node attitudes. |
| $\mathbf{\Theta}$ | An $N \times N$ matrix of differences between attitudes $\mathbf{\theta}$ of the nodes in the network. |

## Dynamics

At initialization, a social network, $\mathcal{G}$, is constructed according to standard network construction algorithms the user can select from. Each node in $\mathcal{G}$ is assigned an attitude sampled from a Gaussian distribution:
$$ \theta_i = \mathcal{N}(0, \sigma) $$
and are constrained to the range $[-\frac{\pi}{2}, \frac{\pi}{2}]$.

The time dynamics of the network's edges and node attitudes are governed by:
$$ A_{ij}'(t) = T_{ij}(t) \sin(\Theta_{ij}(t))$$
$$ \theta_i'(t) = \sum_j T_{ij}(t) A_{ij}(t) \sin(\Theta_{ij}(t)) $$
where $\mathbf{T}(t)$ is an interaction matrix constructing by generating a random matrix, $\mathbf{M}(t)$, with elements in $[0, 1)$ and comparing it to the network's adjacency matrix, $\mathbf{A}(t)$. Where $M_{ij}(t) < A_{ij}(t)$, $T_{ij}(t) = 1$ and nodes $i$ and $j$ interact. Otherwise $T_{ij} = 0$ and nodes $i$ and $j$ don't interact at that time.
where $\Theta_{ij}(t)$ are the elements of the matrix of differences in node attitudes:
$$ \mathbf{\Theta}_{ij}(t) = \mathbf{\theta}(t) - \mathbf{\theta}^{\top}(t) $$

This produces an adaptive complex network where edge strengths and node attitudes are reinforced according to each other over time.
