Spectral Learning of Mixture Models
===================================

Abstract
--------
There has been a series of results recently on recovering hidden parameters
(inference) from probabilistic models using the method of moments. While this
problem is in general hard, use of certain spectral techniques like
eigen-decomposition have resulted in tractable inference using only the first
few moments (3-4) of the data. This article will survey these results, attempt
to describe the approach in general and discuss possible issues with their
generality.


* Problems with Expectation-Maximisation: Convergence and optima
* Problems with Pearson's method: Requires moments of the order of the parameters in the problem.

* Requires some rank assumptions and also availability of multiple views
* Multivariate Gaussians when means are sufficiently separated - $d^c \sigma$.
* Spectral projection

* Note; Non-redundancy is important - else singlarity!

Questions to ask
----------------

* What is SVD doing?
* When NMF > SVD?
* Graphical model = graph - compositional semantics -> polynomial equation?

Reading List
------------

* Vandermonde decompositions of Hankel matrices



[^1]: Anandkumar, Hsu, Kakade; A Method of Moments for Mixture Models and Hidden Markov Models 
