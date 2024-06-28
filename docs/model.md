## Initialization

At initialization, a social network, $\mathcal{G}$, is constructed according to standard network construction algorithms the user can select from. Each node in $\mathcal{G}$ is assigned an attitude sampled from a Gaussian distribution:

$$
\theta_i = \mathcal{N}(0, \sigma)
$$

and are constrained to the range $[-\frac{\pi}{2}, \frac{\pi}{2}]$.

## Dynamics

The time dynamics of the network's edges and node attitudes are governed by:

$$
A_{ij}'(t) = T_{ij}(t) \sin(\Theta_{ij}(t))
$$

$$
\theta_i'(t) = \beta \sum_j T_{ij}(t) A_{ij}^\alpha(t) \sin(\Theta_{ij}(t))
$$

where $\alpha$ and $\beta$ are scaling parameters and $\mathbf{T}(t)$ is an interaction matrix constructing by generating a random matrix, $\mathbf{M}(t)$, with elements in $[0, 1)$ and comparing it to the network's adjacency matrix, $\mathbf{A}(t)$. Where $M_{ij}(t) < A_{ij}(t)$, $T_{ij}(t) = 1$ and nodes $i$ and $j$ interact. Otherwise $T_{ij} = 0$ and nodes $i$ and $j$ don't interact at that time.
where $\Theta_{ij}(t)$ are the elements of the matrix of differences in node attitudes:

$$
\mathbf{\Theta}_{ij}(t) = \mathbf{\theta_i}(t) - \mathbf{\theta_j}(t)
$$

This produces an adaptive complex network where edge strengths and node attitudes are reinforced according to each other over time.

When $\mathbf{A}(t)$ is symmetric, this is a modified Kuramoto model ([Kuramoto, 1975](https://doi.org/10.1007/BFb0013365)), with the modifications:

1. The mean over all phases is replaced by the simple sum.
2. The coupling constant is the adjacency matrix of the network, which may be weighted.
3. The coupling constant has an exponent.

When $\mathbf{A}(t)$ isn't symmetric, this is a circle map ([Ott, 2002](https://www.cambridge.org/core/books/chaos-in-dynamical-systems/7A0749AE3FBBF4312A54D7573C2DAAB5)).
