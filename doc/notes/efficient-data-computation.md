% Efficient Data Computation for Spectral Methods

\newcommand{\Pairs}{\mathrm{Pairs}}
\newcommand{\Triples}{\mathrm{Triples}}

In order to compute moments for "three-view bottleneck" structures in
our graphs, we need to run through our data in multiple passes. This
procedure is particularly tricky (a) if the three-views are not
symmetric (requiring suitable adjustments to be made while estimating
parameters), or (b) the dimension of the problem is large enough that
random projections are needed. This document will show you how you
should be running your computation efficiently.

# Desiderata 

With symmetric views, we need to compute the following two objects,
\begin{align*}
  \hat \Pairs &= \hat \E[x x^T] \\
  \hat \Triples &= \hat \E[x \otimes x \otimes x].
\end{align*}

$\Pairs$ is $d \times d$, and $\Triples$ is $d \times d \times d$,
where $d$ is the dimension of the observables. In language modelling
applications, $d$ is the vocabulary size, about $10^4$; these objects
are too large to explicitly construct.

We will assume that matrix products are efficient to compute for both
$\Pairs$ and $\Triples$. In the language modelling scenario, a matrix
multiply with a $d \times k$ matrix $\Omega$ can be computed in $O(n
\times k)$ time (where $n$ is the number of words being averaged
over),
\begin{align*}
\hat \Pairs \Omega &= \E[ x x^T \Omega ] \\
                   &= \E_{i,j}[ e_i \omega_j^T] \comment{O(n \times k)}\\
\hat \Pairs(\Theta, \Omega) &= \E[ \Theta x x^T \Omega ] \\
                   &= \E_{i,j}[ \theta_i \omega_j^T] \comment{O(n \times k^2)}\\
\hat \Triples(\Theta, \Omega, I)  &= \E[ x x^T \Omega ] \\
                   &= \E_{i,j,k}[ \theta_i \otimes \omega_j \otimes e_k] \comment{O(n \times k^2)}.
\end{align*}

With asymmetric views, 

