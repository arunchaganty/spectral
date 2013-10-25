% Expectation Maximization for the Measurements Model
% Arun Tejasvi Chaganty

\providecommand{\const}{\textrm{constant}}

# Introduction

In this document, I will describe and derive an expectation
maximization algorithm for the measurements model described in
\cite{liang09measurements}.

# Measurements Model

* Let's start by defining the model. The measurements model is
  a discriminative model over output $Y_i$ and measurements $\tau$ and
  is parameterized by $\theta$. $\phi(X_i, Y_i)$ and $\sigma(X_i, Y_i)$
  are arbitrary features. 
  \begin{align}
  p(\theta, Y, \tau | X, \sigma) &\defeq p(\theta) \prod_{i=1}^n p_\theta(Y_i | X_i) p(\tau | X, Y, \sigma) \\
  p(\theta) &\defeq \exp( - \half \eta_\theta \|\theta\|_2^2 + \const ) \\
  p_\theta(Y_i | X_i) &\defeq \exp( \theta^T \phi(X_i, Y_i) - A(\theta;X_i) ) \\
  p(\tau | X, Y, \sigma) &\defeq \exp( - \half \eta_\beta \|\tau - \sum_{i=1}^n \sigma(X_i, Y_i) \|_2^2 + \const ).
  \end{align}
* $\sigma$ is distinguished from $\phi$ in that we also have some
  aggregate statistics $\tau = \sum_{i=1}^n \E[\sigma(X_i, Y_i)] + \textrm{noise}$.
  Thus, conditioning on $\tau$, we get $p(\theta, Y)$.
  \begin{align}
  p(\theta, Y | \tau, X, \sigma) &= \frac{p(\theta, Y \tau | X, \sigma)}{p(\tau | X, \sigma)}. \label{eq:p-theta-y}
  \end{align}

# Expectation Maximization

* One can easily derive that expectation maximization is equivalent to
the alternating minimization objective,
\begin{align}
    \min_\theta \min_{q} &\quad \KL(q(\theta,Y) \| p(\theta,Y | \tau, X, \sigma))\\
    q(\theta', Y) &\defeq \delta_{\theta}(\theta') q(Y).
\end{align}
* Using \equationref{eq:p-theta-y}, we get,
\begin{align}
    \mL(\theta,q) &\defeq  \KL(q(\theta,Y) \| p(\theta,Y | \tau, X, \sigma))\\
    &= -\E_{q(\theta,Y)}[ \log p(\theta, Y | \tau, X, \sigma) ] + \E_{q(\theta,Y)}[ \log q(\theta, Y) ] \\
    &= -\E_{q(Y)}[ \log p(\theta, Y, \tau | X, \sigma) ] + \E_{q(Y)}[ \log q(Y) ] - \underbrace{\log p(\tau | X, \sigma)}_{\const} \\
      &= 
      -\E_{q(Y)}[ \log p(\theta) ] - \sum_{i=1}^n \E_{q(Y)}[p_\theta (Y_i | X_i) ] -\E_{q(Y)}[p(\tau | X, Y, \sigma ) ]
      + \E_{q(Y)}[ \log q(Y) ] + \const \\
      &= 
        \half \eta_\theta \|\theta\|_2^2 
      - \sum_{i=1}^n \theta^T \E_{q(Y)}[\phi(X_i, Y_i)] + A(\theta;X_i) 
      + \half \eta_\beta \E_{q(Y)}[\| \tau - \sum_{i=1}^n \sigma(X_i, Y_i) \|_2^2 ] 
      - H(q(Y))
      + \const.
\end{align}

* Let's scale down the objective with a factor of $\frac{1}{n}$; $\eta_\theta, \eta_\beta$ and $\tau$ absorb a factor of $n$.

\begin{align}
    \mL(\theta,q) 
      &= 
        \half \eta_\theta \|\theta\|_2^2 
      - \frac{1}{n} \sum_{i=1}^n \theta^T \E_{q(Y)}[\phi(X_i, Y_i)] + A(\theta;X_i) 
      + \half \eta_\beta \E_{q(Y)}[\| \tau - \frac{1}{n} \sum_{i=1}^n \sigma(X_i, Y_i) \|_2^2 ] 
      - H(q(Y))
      + \const.
\end{align}

* We make a simple approximation to the objective (so we are no longer precisely doing EM), using Jensen's inequality, 
\begin{align}
  \E_{q(Y)}[\| \tau - \frac{1}{n}\sum_{i=1}^n \sigma(X_i, Y_i) \|_2^2 ] &\ge 
  \| \tau - \frac{1}{n}\sum_{i=1}^n \E_{q(Y)}[\sigma(X_i, Y_i)] \|_2^2. \label{eq:approx}
\end{align}

* Thus, we are minimizing the following objective with respect to $q$ and $\theta$,
\begin{align}
    \min_\theta \min_q &\quad \mL(\theta,q) \\
      &= 
        \half \eta_\theta \|\theta\|_2^2 
      - \frac{1}{n} \sum_{i=1}^n \theta^T \E_{q(Y)}[\phi(X_i, Y_i)] + A(\theta;X_i) 
      + \half \| \tau - \frac{1}{n} \sum_{i=1}^n \E_{q(Y)}[\sigma(X_i, Y_i)] \|_2^2  
      - H(q(Y)).
\end{align}

## E-step

* We are minimizing with respect to $q$. 
* To solve this problem, we can use strong Fenchel duality, $\inf_q \{ f(q) + g(Aq) \} = \sup_\beta \{ -f^* (A^* \beta) - g^*(-\beta)\}$, where $A$ is a linear operator, and $f^*$ is the convex conjugate.
\begin{align}
  f(q) &= -\frac{1}{n}\sum_{i=1}^n \E_q[\log q(Y_i)] - \theta^T (\frac{1}{n} \sum_{i=1}^n\E_q[\phi(X_i,Y_i)]) \\
  A(q) &= \frac{1}{n} \E_q[\sum_{i=1}^n \sigma(X_i,Y_i)] \\
  g(u) &= \half \eta_\beta \|\tau - u\|_2^2 \\
  f^*(A^*\beta) &= \frac{1}{n} \log\int \ud Y ~ \exp(\sum_{i=1}^n \beta^T \sigma(X_i, Y_i) + \sum_{i=1}^n \theta^T \phi(X_i, Y_i) ) \\
         &= \frac{1}{n} \sum_{i=1}^n B(\beta, \theta; X_i)\\
  g^*(-\beta) &= - \beta^T \tau + \half \eta_\beta \|\beta\|_2^2.
\end{align}

Thus, 
\begin{align}
  \min_q \mL(\theta, q) 
    &= 
      \half \eta_\theta \|\theta\|_2^2 + \frac{1}{n} \sum_{i=1}^n A(\theta;X_i) 
      + \min_q 
      - \frac{1}{n} \sum_{i=1}^n \theta^T \E_{q(Y)}[\phi(X_i, Y_i)] 
      + \half \| \tau - \frac{1}{n} \sum_{i=1}^n \E_{q(Y)}[\sigma(X_i, Y_i)] \|_2^2  
      - H(q(Y)) \\
    &= 
       \half \eta_\theta \|\theta\|_2^2 + \frac{1}{n} \sum_{i=1}^n A(\theta;X_i) 
       + \max_\beta - \frac{1}{n} \sum_{i=1}^n B(\beta, \theta; X_i) 
       + \beta^T \tau - \half \eta_\beta \|\beta\|_2^2.
\end{align}

Furthermore, we have that $q^*$ belongs to an exponential family,
\begin{align}
  q_{\beta,\theta}(y | x) &= \exp( \beta^T \sigma(x,y) + \theta^T \phi(x,y) - B(\beta, \theta;x) ).
\end{align}

Now, to solve for $\beta$, we have,
\begin{align}
  \nabla_\beta \mL 
    &= \tau - \eta_\beta \beta - \frac{1}{n} \sum_{i=1}^n \nabla_\beta B(\beta, \theta; X_i) \\
    &= \tau - \eta_\beta \beta - \frac{1}{n} \sum_{i=1}^n \E_{q_{\beta,\theta}(Y)}[ \sigma(X_i, Y_i) ].
\end{align}

## M-step

* In the M-step, minimize with respect to $\theta$, using the optimal $q$ found in the E-step. 
\begin{align}
 \min_\theta \mL(\theta,q) &= 
    \half \| \tau - \frac{1}{n}\sum_{i=1}^n \E_{q^*(Y)}[\sigma(X_i, Y_i)] \|_2^2  
    - H(q^*(Y)) \\
    &\quad + \min_\theta 
    \half \eta_\theta \|\theta\|_2^2 + \frac{1}{n} \sum_{i=1}^n A(\theta;X_i) 
    - \frac{1}{n} \sum_{i=1}^n \theta^T \E_{q^*(Y)}[\phi(X_i, Y_i)]  \\
  \theta^* 
      &=
      \arg\min_\theta 
        \half \eta_\theta \|\theta\|_2^2 + \frac{1}{n} \sum_{i=1}^n A(\theta;X_i) 
        - \frac{1}{n}\sum_{i=1}^n \theta^T \E_{q^*(Y)}[\phi(X_i, Y_i)].
\end{align}
* To solve for $\theta$, we get,
\begin{align}
  \nabla_{\theta} \mL(\theta,q) 
    &= \eta_\theta \theta + \frac{1}{n}\sum_{i=1}^n \nabla_\theta A(\theta;X_i) 
        - \frac{1}{n}\sum_{i=1}^n \E_{q^*(Y)}[\phi(X_i, Y_i)]  \\
        &= \eta_\theta \theta + \frac{1}{n}\sum_{i=1}^n \E_{p_\theta(Y)}[ \phi(X_i,Y_i) ]
        - \frac{1}{n}\sum_{i=1}^n \E_{q^*(Y)}[\phi(X_i, Y_i)].
\end{align}

# Notes

* Because of the approximation in the \equationref{eq:approx}, we no
longer have a guarantee that the solution $\theta^*$ is a local maxima
of the likelihood.

