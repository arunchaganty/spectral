"""
Random Distributions particular to spectral algorithms
"""

import scipy as sc
import scipy.stats as stats
import scipy.linalg as linalg

def shuffle_permutation( n ):
    """Generate a random permutation as a sequence of swaps"""
    n = int( n )
    return sc.array( map( sc.random.randint, xrange( n-1, 0, -1) ) )

def permutation( n ):
    """Generate a random permutation as a sequence of swaps"""
    n = int( n )
    lst = sc.arange( n )
    sc.random.shuffle( lst )
    return lst

def matrix_permutation( m, n ):
    """Generate a random matrix permutations"""
    m, n = int( m ), int( n )
    return permutation( m ), permutation( n )

def orthogonal(n):
    """Generate a random orthogonal 'd' dimensional matrix, using the
    the technique described in: 
    Francesco Mezzadri, "How to generate random matrices from the
    classical compact groups" 
    """
    n = int( n )
    z = sc.randn(n, n) 
    q,r = sc.linalg.qr(z) 
    d = sc.diagonal(r) 
    ph = d/sc.absolute(d) 
    q = sc.multiply(q, ph, q) 
    return q

def wishart(n, V, nsamples=1):
    """wishart: Sample a matrix from a Wishart distribution given
    by a shape paramter n and a scale matrix V
    Based on: W. B. Smith and R. R. Hocking, Algorithm AS 53: Wishart
    Variate Generator, Applied Statistic, 21, 341

    Under the GPL License
    From the Astrometry project: http://astrometry.net/

    W(W|n,V) = |W|^([n-1-p]/2) exp(-Tr[V^(-1)W]/2)/ ( 2^(np/2) |V|^(n/2)
    pi^(p(p-1)/2) Prod_{j=1}^p \Gamma([n+1-j]/2) )
    where p is the dimension of V

    Input:
       n        - shape parameter (> p-1)
       V        - scale matrix
       nsamples - (optional) number of samples desired (if != 1 a list is returned)

    Output:
       a sample of the distribution

    Dependencies:
       scipy
       scipy.stats.chi2
       scipy.stats.norm
       scipy.linalg.cholesky
       math.sqrt

    History:
       2009-05-20 - Written Bovy (NYU)
    """
    #Check that n > p-1
    p = V.shape[0]
    if n < p-1:
        return -1
    #First lower Cholesky of V
    L = linalg.cholesky(V,lower=True)
    if nsamples > 1:
        out = []
    for kk in range(nsamples):
        #Generate the lower triangular A such that a_ii = (\chi2_(n-i+2))^{1/2} and a_{ij} ~ N(0,1) for j < i (i 1-based)
        A = sc.zeros((p,p))
        for ii in range(p):
            A[ii,ii] = sc.sqrt(stats.chi2.rvs(n-ii+2))
            for jj in range(ii):
                A[ii,jj] = stats.norsc.rvs()
        #Compute the sample X = L A A\T L\T
        thissample = sc.dot(L,A)
        thissample = sc.dot(thissample,thissample.transpose())
        if nsamples == 1:
            return thissample
        else:
            out.append(thissample)
    return out

