% Spectral Experts
% Arun Chaganty, Percy Liang

# Introduction

* Latent variable models (HMM, mixture models, CRFs, etc.) are extremely powerful models of data we've developed.
* However, learning parameters in these models is typically hard because their likelihood functions are not convex in the parameters.
* Most common approach is to use local methods like EM, variational techniques, etc.
* Local optima are a serious concern when using these methods. 
* Importantly, this is a problem that does not necessarily go away with more and more data!
* Recently, there have been a number of _consistent estimators_ proposed based on the method of moments.
  + These are called "spectral methods" for how they utilize spectral decompositions to recover the parameters.
* Our work extends this approach of learning parameters to the discriminative setting wherein the moments of the parameters are not directly observed.
  + The crux of our approach will be to use regression to first learn these moments, followed by application of tensor decomposition to learn the parameters.

# Background

## Method of Moments

* To start with, let us study how method of moment estimators work in general.
* Consider a moment map $\mathcal{M}$ that maps the parameters to the moments. For a Gaussian, we have that $\mathcal{M} = (\mu, \sigma^2)$.
* By the central limit theorem, our sample estimates of the moments converge at a $1/\sqrt{t}$ rate.
* We will take the inverse of the moment map on the sample estimates to learn the parameters.
* We also know that if the moment map is bijective and differentiable, the inverse will also converge at the same $1\sqrt{t}$ rate.

# Algorithm

## Recovering the moments

* The first problem we run into is that we can't observe the moments of $\beta$ directly!
* However, observe that $y = \innerp{\beta_h}{x} + \epsilon$. \todo{Write rest of the equation}.
* Note that $\eta_1$ is a bounded-zero mean noise term: by regressing on $(y,x)$, we can learn $M_1$. 
* However, the first moments are insufficient to learn this model.
* Let's look at the second and third moments. \todo{Look at the 2nd and 3rd moments.}
* Further, we can exploit the low rank structure of $M_2$ to use low-rank regression here.
* Finally, this approach gives us a consistent estimator of the moments.

## Recover the parameters

* Now that we've the moments, let us review how the tensor decomposition technique can be used to learn the parameters.
* $M_2$ and $M_3$. 
* Observe that the decompositions share a basis.
* By exploiting tensor decomposition, we can find what these are.
* \todo{Describe the tensor decomposition algorithm, briefly.}

# Theorem: Rates of Recovery

* We can divide the rates for the two parts; learning the moments themselves, learning the parameters.
* In the first case, describe, the tensor recovery theorem (Tomioka2011).
  + Diagram breaking the error into two parts, bounding $k(X)$ and $\|X^*\|$.
* Tensor recovery
  + Diagram breaking the error into a chain, adding factors.
* Present the complete bound.
  + A minute to describe what are the large order dependences.

# Spectral Experts in Practice

* We simulated the performance of spectral experts. 
  + Follow the one example.
  + Even with $O(10^5)$ samples, the parameters recovered aren't excellent.
* Motivate initialization with the energy landscape picture.
* Other experiments.
  + Be frank, this isn't going to replace EM on it's own, but perhaps a principled approach to initializing EM?

