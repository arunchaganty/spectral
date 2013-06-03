% Spectral Experts
% Arun Chaganty, Percy Liang

\newcommand{\tp}[1]{^{\otimes #1}}
\newcommand{\opX}{\mathfrak{X}}
\newcommand{\cvec}{\textrm{cvec}}

# Introduction

* Latent variable models (HMM, mixture models, CRFs, etc.) are extremely powerful models of data we've developed.
* However, learning parameters in these models is typically hard because their likelihood functions are not convex in the parameters.
* Most common approach is to use local methods like EM, variational techniques, etc.
* Local optima are a serious concern when using these methods. 
* Importantly, this is a problem that does not necessarily go away with more and more data!
* Recently, there have been a number of _consistent estimators_ proposed based on the method of moments \todo{Include citations}.
    + These are called "spectral methods" for how they utilize spectral decompositions to recover the parameters.
* Our work extends this approach of learning parameters to the discriminative setting wherein the moments of the parameters are not directly observed.
    + The crux of our approach will be to use regression to first learn these moments, followed by application of tensor decomposition to learn the parameters.

# Background

## Mixture of Regressions

* The mixture of linear regressions model defines a conditional distribution over a response $y \in \Re$ given covariates $x \in \Re^d$.
* The generative procedure is as follows,
    i) Draw a mixture component $h \in [k] \sim Mult(\pi)$, where $\pi = [\pi_1 | ... | \pi_k]$ defines the mixture proportions.
    ii) Draw the noise $\epsilon \sim \mathcal{E}$, where $\mathcal{E}$ is the noise distribution.
    iii) Set $y = \beta_h^T x + \epsilon$, where $\{\beta_h\}_{h=1}^{k}$ are the conditional means of the regression coefficients.
* The parameters that we would like to learn from this model are $\pi$ and $B = [ \beta_1 | ... | \beta_k ]$.
* The challenge in our scenario is that the moments of the data give us very filtered information of the parameters.

## Method of Moments

* Let us study how method of moment estimators work in general.
* Consider a moment map $\mathcal{M}$ that maps the parameters $\theta$ to the moments $m$. For a Gaussian, we have that $\mathcal{M} = (\mu, \sigma^2)$.
* In general, we will compute the inverse of the moment map to learn the parameters from the sample estimates.
* By the central limit theorem, our sample estimates of the moments converge at a $1/\sqrt{n}$ rate, so we expect that our parameters will also converge at this rate.


# Algorithm

## Recovering the moments

* As noted earlier, the first problem we run into is that we can't observe the moments of the parameters $B$ and $\pi$ directly!
* However, observe that 
\begin{align}
  y &= \innerp{\beta_h}{x} + \epsilon \\
    &= \innerp{M_1}{x} + \underbrace{ {\beta_h - M_1}{x} + \epsilon }_{\textrm{noise}},
\end{align}
  where $M_1 = \sum_{h=1}^k \pi_h \beta_h$, the mean regression coefficent. We note that while the noise term is dependent on $x$, it has a zero-mean. Thus, we could potentially recover $M_1$ through regression.
* However, the first moments are insufficient to learn this model.
* Let's look at the second and third moments of the data, 
\begin{align*}
  y^2 &= (\innerp{\beta_h}{x} + \epsilon)^2 \\
    &= \innerp{M_2}{x\tp{2}} + \E[\epsilon^2] +
     \underbrace{ \innerp{\beta_h\tp{2} - M_2}{x\tp{2}} + \innerp{\beta_h\tp{1} - M_1}{x\tp{1}} + (\epsilon^2 - \E[\epsilon^2]) }_{\textrm{noise}}, \\
  y^3 &= (\innerp{\beta_h}{x} + \epsilon)^3 \\
    &= \innerp{M_3}{x\tp{3}} + 3\E[\epsilon^2] \innerp{M_1}{x}  + \E[\epsilon^3] + {\textrm{noise}},
\end{align*}
* On it's own, $M_2$ is insufficient to identify the model, because it is invariant to rotations of $B$. 
* However, it turns out that $M_2$ and $M_3$ are sufficient to identify the model, via tensor decomposition, if $k < d$.
* An additional fact that we can exploit is that both $M_2$ and $M_3$ are low rank, so we can use low-rank regression to recover estimates $\hat M_2$ and $\hat M_3$ efficiently from data.

### Caveat: Requirements for Identifiability

* In ordinary linear regression, the regression coefficients $\beta \in
\Re^d$ are identifiable if and only if the data has full rank:
$\E[x\tp{2}] \succ 0$.
* However, because we need regression on higher moments to recover $M_2$ and $M_3$, we also need that 
$\E[\cvec(x\tp{p})\tp{2}] \succ 0$ for $p \in \{1,2,3\}$.
* This has some subtle implications when the features are completely independent of each other.
* For example, if $x = (1, t, t^2)$, then $\E[\cvec(x\tp{2})\tp{2}]$ is singular for any data distribution.

## Recovering the parameters

* Now that we've the moments, let us review how the tensor decomposition technique can be used to learn the parameters.
* The method exploits the fact that $M_2$ and $M_3$ share a basis, 
  \begin{align*}
    M_2 &= \sum_{h=1}^k \pi_h \beta_h\tp{2} \\
    M_3 &= \sum_{h=1}^k \pi_h \beta_h\tp{3}.
  \end{align*}
* Decompositions are not unique however, but we can use the whitening transformation for $M_2$ to give _whiten_ $M_3$ such that it has a orthogonal decomposition.
  \begin{align*}
    I &= W^T M_2 W  \\
      &= \sum_{h=1}^k (\underbrace{\sqrt{\pi_h} W^T \beta_h}_{v_h})\tp{2} \\
    M_3(W,W,W) &= \sum_{h=1}^k \pi_h (W^T \beta_h)\tp{3} \\
               &= \sum_{h=1}^k \frac{1}{\sqrt{\pi_h}} v_h\tp{3}.
  \end{align*}
* The robust tensor power method by \cite{AnandkumarHsuGe2012} find stable eigenvectors using the following iterative algorithm,
$v_h \to \frac{T(I, v_h, v_h)}{||T(I, v_h, v_h)||_2}.$

# Theorem: Rates of Recovery

* The rate of convergence for the spectral experts algorithm to the
true parameters breaks into two parts; the rates for learning the
moments, which feeds into the rates for learning the parameters.
* \todo{Diagram showing how the error breaks down.}
* For low rank regression, we have the following bound on recovery by (Tomioka2011);
  $$ || \hat M_p - M_p ||_F \le \frac{32 \lambda^{(p)}_n \sqrt{k}}{\kappa(\opX_p)}, $$
where $\kappa(\opX_p)$ is the (restricted) strong convexity constant, and $\lambda^{(p)} > ||\opX^*_p(\eta)||$. 
* Because we assume our noise is bounded, it is easy to show that the error concentrates. 
* In the tensor recovery case, we will need to whiten $M_3$ before
applying the tensor decomposition and unwhiten it afterwards; this
modifies the error bounds slightly.

# Spectral Experts in Practice

* We simulated the performance of spectral experts and compared it to EM. 
    + We generated data from the mixtures of linear regressions model, with $x$ drawn uniformly in $[-1,1]^d$.
* In a non-trivial number of cases, EM did not converge to the right parameters and got stuck in local optima.
* As a specific instance, we studied this example, \todo{Include diagram}.
    + Only 13 of a 100 attempts with EM successfully identified the true parameters.
    + On the other hand, even with $O(10^5)$ samples, the parameters by the spectral method weren't great.
    + However, EM when initialized with these parameters did extremely well. 
* This finding that spectral methods should be a good initialization for EM is not surprising.
    + The biggest sell for spectral methods is that they give a global guarantee on where the parameters are.
    + The parameters might not be at the global optima, but will hopefully lie in the potential well around it.
    + EM will then converge to the global optima.
    + \todo{Include diagram.}
* \todo{Summarize Other experiments.}
    + Be frank, this isn't going to replace EM on it's own, but perhaps it highlights a principled approach to initializing EM.

# Conclusions

* We can learn the parameters of conditional models where the observed moments on their own have such sparse information.
* The key intuition is that we could construct a regression problem to learn the moments of the parameters from their projections onto the data.
* We found that while the parameters learned this way are usually not better than those learned via local methods like EM, etc., they are a good way to initialize these local methods.

