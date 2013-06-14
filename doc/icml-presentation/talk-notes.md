% Talk Transcript

# Preamble (0:14 = 0:14) [0:15]

* Consistent parameter estimation for a simple discriminative model, the mixture of linear regressions.

# Generative vs. Discriminative Latent Variable Models (1:20 = 1:34) [1:00]

* Latent variable models are very powerful tools in machine learning. 
* We can categorize these models into two broad classes; generative and discriminative. 
* Generative models describe how observed variables, $x$, are generated conditioned on latent variables, $h$. 
* Gaussian mixture models, hidden Markov models, and so on are examples of this class.

* In contrast, in discriminative models describe how an output $y$ is generated from some input $x$, conditioned on the latent variables. 
* Mixture of experts, latent CRFs, discriminative LDA are examples in this class.
* Lot of prior work by in addressing consistent parameter estimation for the generative family, but not much work has been done in the discriminative setting. 
* Important because discriminative models are amenable to including arbitrary features and tend to be more accurate.
* Direction with this work.

# Parameter estimation is hard (1:34 = 3:06) [1:20]

* Parameter estimation is in general hard because the latent variables introduce a non-convexity in the (negative) likelihood function. 
* MLE is a consistent estimator; 
  + true parameters, $\theta^*$ in the limit of infinite data. 
  + Usually intractable.
* In practice, we often use a local estimator, like EM or a gradient based method. 
  + computationally tractable, but sensitive on initialization. 
  + We can get stuck in local optima, like the $\hat \theta$; such local optima often occur in practice. 
* Question we ask:
  + Can we build a efficient and consistent estimator? One that will have a polynomial computational and sample complexity.
  + Our approach is to use the method of moments.

# Method of Moments / Related Work (0:54 = 4:10) [0:45]

* Method of moments is an old technique, starting with Pearson in 1894.
* One line of work focusses on recovering observable operators that are useful at prediction, but do not give us parameters. 
  + This work found initial applications in control theory, but recently, there have been several applications in machine learning for example, the hidden Markov model by Hsu, Kakade and Zhang in 2009.
* We build on another line of work focussed on parameter estimation. 
  + In particular, we are influenced by the treatment of gaussian mixture models, by Anandkumar, Hsu and Kakade in 2012.
    
* Going ahead, 
  + review tensor factorization by Anandkumar et al., 2012 
  + apply this idea to our discriminative setting.

# Aside: Tensor Operations (0:54 = 5:06) [0:30]

* First, I'd like to review some basic tensor operations. 
  + $x\tp{3}$ is $x$ outer product x outer product x, call $x$ cubed.
  + Tensor whose $(i,j,k)$-th component is $x_i x_j x_k$.
* Next, I'll use the angular brackets for a generalized inner product.
  + Element-wise product and sum.
  + Think of it as vectorize the tensors and taking a dot product.

# Example: Gaussian Mixture Model (2:44 = 7:50) [2:00]

* Generative process for the GMM
* There are $K$ clusters.
* $h$, which chooses the mixture component. 
* For a $h$, we choose a mean $\beta_h$ and draw a Gaussian centered around $\beta_h$.
* Goal is to recover $\pi$, $B$.

* Cluster centers (not cond. means) are $\beta_h$, but we don't observe $h$.
* Weighted sum of mean, but insufficient.
* Weighted sum of each mean squared; not identifiable (no details).
* Weighted sum of each mean cubed.

* Key idea presented in Anandkumar 2012 is that cubed are factorized and if orthogonal, they are eigenvectors.
* In general, we can whiten the tensor using the second moments.

# Interlude: Discriminative vs. Generative (0:25 = 8:15) [0:20]

* In the beginning, I talked about two classes of LVMS.
* We looked at how the tensor factorization could be exploited in a generative model.
* We are going to extend this to discriminative, with MLR.

# Mixture of Linear of Regressions (1:00 = 9:15) [1:00]

* Let's look at the model. Data generated from lines.
* Pick a line, pick a point.
* In the data we're given, we don't observe lines.

+ Our objective: given data, can we recover parameters.

# Finding Tensor (2:35 = 11:50) [2:30-3:00]

* We are going to exploit the tensor factorization, but first we need to extract this tensor from our data.
* Let's _mediate_ on the equation for $y$.
* Observation noise.
* $\beta_h$ is random.
* Linear measurement, noise.
  + Noise = mixing noise plus observation noise. 
  + Mixing noise depends on $x$.
* Can we extend to higher powers of $\beta$ and recover $\beta^3$?

* Let's look at $y^2$.
  + Inner product between $\beta\tp{2}$ and $x\tp{2}$.
  + known bias
  + Call this $M_2$.
* Next, $y^3$.

* Now we have tensor $M_3$, we note that it has the same factorization structure.
* We can apply the tensor factorization algo.

# Algorithm overview (1:15 = 13:05) [1:30]

* We have all the pieces to recover parameters.
  + Start with regression on powers of the data
  + Then do tensor factorization.

{Talk about the details: assumptions}

For regression to work, we need the assumption that $\E[\beta_h\tp{p}]$ has rank at least $k$, and for the tensor factorization step, we need that $k < d$ and that $B$ is full rank.

# Low rank

* Doing regression on these higher powers, we're doing regression with $d^2$ and $d^3$ dimensions, 
* underlying dimensionality is just $kd$. 
* Exploit using nuclear norm regularization.
* Extended to tensors.

# Sample Complexity

* Both have polynomial computational complexity.
* Explain $x^12$

# Experimental insights (2:45 = 16:50) [2:30]

* Some insights we derived from simulated experiments.
* Consider this example; inputs are a polynomial terms of $t$.
* parameters $k=3, d=4, n=10^5$.
* Ran EM - stuck in local optima. Explain how.
* Over 200 initializations; parameter error. 
* Spectral, it did better than most EM, but not the best. "Doesn't hug the data, but right ball park."
* EM initialized with spectral.
* IMPORTANT: don't say that we don't recover the true parameters - be explicit about recovering in the limit.
* We observed similar performance on simulated examples for several different $d$ and $k$.
* These graphs plot parameter error averaged over 10 different examples with the particular $d$ and $k$, and over 10 different initializations each. 

# On Initialization

* Fit intuition painted early on.
* EM can get stuck in a local optima but fit the data well.
* Spectral, all we can say is we're in some interval epsilon; could not fit exactly.
* EM from there will fit well.

# Conclusions (0:45 = 17:35) [0:45]

* In conclusion, I've presented a consistent estimator for the mixture of linear regressions.
* The key idea was to expose the tensor structure through regression.
* We showed that it indeed has polynomial sample and computational complexity.
* Empirically, we found good initialization.
* Going forward, we'd like to see how we could handle other disc. models.
* In particular, how can we handle dependencies between $x$ and $h$? Allows us to model MoE.
* Another direction we think is important is handling dependencies non-linear link functions like in logistic regression.

# Thank you.

# Question: Isn't it better to run EM with many random initializations then use your method?

We found that the number of initializations that led to the true parameters were fairly small even on our toy data; it's unclear how many initializations you would need to reasonable find good parameters for larger settings. The spectral method is pretty efficient and in our experiments, took on the same order of time as running full EM, so I don't think there is a downside to using the method, provided you have enough sufficient data.

# Question: In principle your method is supposed to converge. Does it?

The convergence results we present only hold when we have sufficient samples for some conditions to hold; these conditions depend on the condition number of the powers of $\beta$. On experiments without any featurization, the condition numbers were usually quite reasonable and we observed convergence. On the polynomial featurizations we experimented with, we found the condition numbers to be much worse and it appeared we required more data than we could run our code for to observe convergence.

# Question: Isn't it of concern that your method doesn't scale well for very large $d$? 

The primary regime for the method is when $k << d$ but making this method work better with larger $d$ is an important next step.


