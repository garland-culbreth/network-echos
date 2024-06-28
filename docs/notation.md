# Notation

| Variable | Definition |
|---|---|
| $t$ | Time. |
| $N$ | Number of nodes in the network. |
| $\mathbf{A}$ | The $N \times N$ adjacency matrix of the social network. A discrete function of time. Entries are the weights of the edges connecting the $i$-th and $j$-th nodes. |
| $\mathbf{T}$ | An $N \times N$ interaction matrix. Entries are $1$ or $0$ denoting whether nodes $i$ and $j$ interact at time $t$. |
| $\mathbf{M}$ | An $N \times N$ matrix of uniform random numbers between $0$ and $1$. Used to determine which nodes interact at time $t$. |
| $\mathbf{\theta}$ | A length $N$ column vector of node attitudes. |
| $\mathbf{\Theta}$ | An $N \times N$ matrix of differences between attitudes $\mathbf{\theta}$ of the nodes in the network. |
| $\mathbf{\alpha}$ | Scaling parameter governing the dependence of attitude reinforcement on the adjacency matrix. This is a complexity parameter. |
| $\mathbf{\beta}$ | Scaling parameter governing the rate of attitude change. Smaller values slow change. |
