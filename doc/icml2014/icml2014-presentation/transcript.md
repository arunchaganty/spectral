% Talk Transcript

# Preamble (0:14 = 0:14) [0:15]

Hello. I am going to talk about consistent parameter estimation for a simple discriminative model, the mixture of linear regressions. This is in contrast to a lot of prior work in learning generative models in the spectral learning community.

# Generative vs. Discriminative Latent Variable Models (1:20 = 1:34) [1:00]

Latent variable models are very powerful tools in machine learning, but are hard to learn because the latent variables introduce a non-convexity in the likelihood function. We can categorize these models into two broad classes; generative and discriminative. 

In the first case of generative models, we describe the generative distribution for the observed variables, $x$, conditioned on some latent variables, $h$. 

Gaussian mixture models, hidden Markov models, and so on are examples of this class.

In contrast, in discriminative models the generative distribution for some observed variables, for example the class labels, $y$, are described conditioned not only on the latent variables, but also the other observed variables, $x$. 

Latent CRFs, discriminative LDA, the mixture of experts, etc. are examples in this class.

There has been a lot of prior work by the spectral learning community in addressing consistent parameter estimation for the generative family (citations?), but not much work has been done in the discriminative setting. 

We think this is important because discriminative models are often preferred in practice because they make it easier to include arbitrary features and tend to be more accurate. 

{Highlight box}

# Parameter estimation is hard (1:34 = 3:06) [1:20]

{Draw axes in the cartoon}

Parameter estimation for latent variable models is in general hard. One of the main culprits is that the log-likelihood functions in these models are non-convex. 

The maximum likelihood estimator will get us the true parameters, $\theta^*$ in the limit of infinite data. In fact, statistically, it is the best estimator for a given amount of data, but it is in general intractable.

In practice, we might use a local estimator, like EM or a gradient based method. This is computationally tractable, but can be very dependent on our initialization. With the right initialization, we might find the true parameters, but for some other initialization we could get stuck in local optima, like the $\hat \theta$ shown here. Such local optima often occur in practice. 

Our goal, like that many other spectral methods, is to develop an algorithm with polynomial sample and computational complexity, and we will do so using the method of moments. In the limit of infinite data, we are guaranteed to recover the true parameters. Moreover, with finite samples, we can still guarantee our estimates to be within $\epsilon$ of the true parameters. 

# Method of Moments / Related Work (0:54 = 4:10) [0:45]

The method of moments itself has a rich history. It was first introduced by Pearson in 1894, and found applications in control theory. 

Closer to home, there have been several applications in the observable operator space, for example, predictive state representations, and more recently for the hidden Markov model by Daniel Hsu et al. in 2009.

We build on a line of work that is focussed on parameter estimation. In particular, we are influenced by the treatment of multi-view mixture models, by Anandkumar et al., which was published recently in 2012. I'd like to review the tensor factorization technique used by Anandkumar et al. which can be used to recover parameters for the multi-view mixture model; we'll shortly see how it can be applied to our discriminative model.

# Aside: Tensor Operations (0:54 = 5:06) [0:30]

Before I can begin, I'd like to introduce some notation for the tensor operations I'll be using.
I'll use $\otimes$ to refer to the outer or tensor product. $x\tp{3}$, which I'll just call $x$ cubed, is a tensor whose $(i,j,k)$-th component is $x_i x_j x_k$.
Next, I'll use the angular brackets for a generalized inner product.
Finally, I'll consider projections $A(u)$ for the matrix formed by summing out the third-index dot-producted with the vector $u$. 
We can also project along two axes of the tensor to get a vector.

# Example: Multi-view Gaussian Mixture Model (2:44 = 7:50) [2:00]

{Use a two-column format instead} 
Alright, let's consider the generative process for the multi-view mixture, with the assumption that $k \le d$. We have a single latent variable, $h$, which chooses the mixture component, (one of these three). For each mixture component, we'll generate three views, $x_1, x_2, x_3$ that, for simplicity, are identically distributed as Gaussian with mean $\beta_h$. 

Let's write out the moments for this model. Note that the conditional means are the $\beta_h$ by definition. Now the first moment of the data is the weighted average of the component means. Clearly just observing this is insufficient to identify our parameters.

Next, we'll look at the cross moments of $x_1$ and $x_2$; the cross moments lets us ignore variance terms since $x_1$ and $x_2$ are independent, and hence this is just composed of $K$ rank-one matrices. 

The second moments aren't also insufficient to identify the parameters though; an orthogonal transformation of the $\beta_h$ would give us the same moments.
Finally, let's look at the third order moments for this model; it's similarly composed of $K$ rank-one third-order tensors. 

{Swap image and eigenvector notation} 
The key idea is that if component means $\beta_h$ were orthogonal, then they are _eigenvectors_ of the third order tensor; if we projected this tensor along $\beta_h$ twice, we get $\pi_h \beta_h$. We can use the tensor power method to recover the eigenvectors from the third moments. 

More generally, we can use the second order moments to whiten the third order tensor; for details, I refer you to a prepint by Anandkumar, Hsu, et al. (2013).

# Interlude: Discriminative vs. Generative (0:25 = 8:15) [0:20]

{Improve slide}

In the beginning of this talk, I described two broad classes of latent variable models, generative and discriminative, and motivated why we would like to develop a consistent estimator for the latter class. I will now introduce the particular instance that we developed an algorithm for, the mixture of linear regressions.

# Mixture of Linear of Regressions (1:00 = 9:15) [1:00]

Again, let's start with the generative process for this model. We'll assume that the $h$ and $x$ are independent of each other. The mixture of linear regressions models our data as a mixtures of lines plus noise. 

For any given $x$, we'll select a single line based on the latent variable $h$. Then, we'll observe the coordinate $y$ along the line at that $x$, but maybe with some noise. For technical reasons, we'll assume that the noise is bounded. 

For another $x$, we pick a line at random again, and then observe the coordinate $y$ with some noise. We can repeat this process till we get a sizable dataset.

Our objective is then, given this data, can we develop a procedure to recover the parameters; the mixture probabilities, $\pi$, and the set of regression coefficients, the slopes of the lines $B = [\beta_1| ... | \beta_h]$?

# Method of Moments for the Mixture of Linear Regressions (2:35 = 11:50) [2:30-3:00]

Our plan is to reduce the problem to the method of moments, where we observed the moments of the data. Can we get something similar?

{Include the messy moments picture?}

Let's meditate a bit on the generative process. This process looks awfully like conventional regression, but we have to be careful; the $\beta_h$ are actually random variables. It's not immediately obvious what we would get if did do regression here.

So, to make it obvious, let's add and subtract the expected value of the $\beta$s, with respect to $h$. Now, we have two terms; a constant vector dot the data plus some zero-mean term, which we can consider to be "noise".
What this tells us is that if we did regression here, we would get the expected value of the regression coefficients.

Can we generalize this process to get the expected value of higher powers of the parameters $\beta_h$? 

Here, I've replaced the inner product with this angular bracket notation, which, remember, was the generalized inner product.

{Include manipulation?}

Now, if we wrote out the expression for the second power of $y$, and did a little bit of arithmetic manipulation, we're going to see the same structure; an inner product of the expected value of the second power of the $\beta$s and the second power of the data $x$, plus some noise and this time, we also have some known bias. This means we can use regression here to get the second powers of $\beta$s.

We can extend this approach to the third power and so on. 

{Include this structure picture on a slide}

Now, let's look at the structure of the third power of $\beta$; it's identical to the third moment of the data in the multi-view mixture case. That means we can use the tensor factorization algorithm here too! 

# Algorithm overview (1:15 = 13:05) [1:30]

{Move to the algorithm overview}

So that's the rough skeleton of the algorithm, regression to get these powers and tensor factorization to get the parameters from there.

{Talk about the assumptions}

For regression to work, we need the assumption that $\E[\beta_h\tp{p}]$ has rank at least $k$, and for the tensor factorization step, we need that $k < d$ and that $B$ is full rank.

{Return to the regression picture}

We have a statistically consistent estimator now, since each step is consistent, but doing vanilla regression on these higher powers could potentially be very statistically inefficient. We know that we need $O(d^3)$ samples to do regression, but we should be able to exploit the fact that even for $\beta_h\tp{3}$, there are only $kd + k$ degrees of freedom; we know that these matrices and tensors have a rank of only $k$. To exploit this, we can use low rank-regression, which adds a nuclear norm regularizer to the optimization objective. 

{return to algorithm overview}

This is our complete algorithm. As you can see, each part has a polynomial computational complexity, so we have a polynomial computational complexity.

# Perturbation Analysis (1:00 = 14:05) [1:00]

{Fix horrendous picture}
Now, I'll briefly talk about the sample complexity of our algorithm. We adapt results for low-rank regression from Negahban and Wainwright (2009) and Tomioka (2012) to our noise setting. We can show that we need a number of samples polynomial in all our parameters to get estimates of the power of $\beta_h$ with an $\epsilon$ error. 

Next, we apply the analysis of the tensor factorization algorithm to give us bounds on the recover of the parameters in terms of error of the inputs, $\beta_h^3$. This step also requires just polynomial number of samples to get the parameters within $\epsilon$ of the true value. 

Thus, our algorithm has a polynomial sample complexity as well.

# Experimental insights (2:45 = 16:50) [2:30]

{Fix layout}
Finally, I'd like to talk about some insights we found on simulated experiments. We can consider featurizations of our input data; so in this example, where with data $x$ and response $y$, we consider a polynomial featurization that gave use these curves. 

The dimension of our data $d$ is $4$, and the number of components $k$ is $3$. For our experiments, we used $10,000$ examples generated from the model.

When we ran EM with random initializations on this algorithm, we found that it easily got stuck in a local optima. This in fact occurred on a large percentage of the random initializations we tried. The histogram counts the number of initializations out of 200 random initializations which achieve a particular parameter error. In this example, only about 13\% actually converged to the true parameters. 

Next, we tried the spectral method. We didn't exactly recover the parameters, but we are operating in the finite sample regime and this can occur. Importantly though, it seems like the parameters we recover sufficiently separate the different mixture components. We tried EM initialized with these parameters, and find that we recover the true parameters with much higher frequency.

{Include table}
We ran similar experiments on a number of simulated datasets with a different number of dimensions and components and generally observed this trend. I've presented the parameter errors averaged over 10 different datasets within each configuration of $d$ and $k$, and over 10 initializations for each dataset.

{Include parameter annotations}
These observations fit with the intuition painted by the log-likelihood picture we began with. Though the log-likelihood function is non-convex, the method of moments estimator guarantees that we'll recover a parameter estimate within $\epsilon$ of the true parameters. This might have a lower likelihood than EM, but if we can get $\epsilon$ small enough, it can be in the right ball park of the true parameters that running EM initialized with these parameters will converge to the true parameters.

# Conclusions (0:45 = 17:35) [0:45]

In conclusion, I've presented a consistent estimator we developed for the mixture of linear regressions with a polynomial sample and computational complexity.

The key idea was that we can use regression to recover the powers of $\beta_h$ which could then be factorized using the tensor factorization technique.

Finally, our experiments support the intuition that in the finite sample regime, method of moment estimators can be good initialization for EM.

# Thank you.

# Question: Isn't it better to run EM with many random initializations then use your method?

We found that the number of initializations that led to the true parameters were fairly small even on our toy data; it's unclear how many initializations you would need to reasonable find good parameters for larger settings. The spectral method is pretty efficient and in our experiments, took on the same order of time as running full EM, so I don't think there is a downside to using the method, provided you have enough sufficient data.

# Question: In principle your method is supposed to converge. Does it?

The convergence results we present only hold when we have sufficient samples for some conditions to hold; these conditions depend on the condition number of the powers of $\beta$. On experiments without any featurization, the condition numbers were usually quite reasonable and we observed convergence. On the polynomial featurizations we experimented with, we found the condition numbers to be much worse and it appeared we required more data than we could run our code for to observe convergence.

# Question: Isn't it of concern that your method doesn't scale well for very large $d$? 

The primary regime for the method is when $k << d$ but making this method work better with larger $d$ is an important next step.

