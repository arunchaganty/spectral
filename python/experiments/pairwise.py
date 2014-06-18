#!/usr/bin/env python2.7
"""
A simple script to compare composite likelihood and pseudo-inversion based estimators.
"""

import ipdb

import numpy as np
from numpy.linalg import inv, svd, pinv, norm
from numpy.random import rand, randn
from numpy import zeros, ones, eye
import itertools as it

from collections import Counter

np.random.seed(12)

def generate_random_parameters(k, d):
    pi = normalize(rand(k))
    O = column_normalize(rand(d, k))
    T = column_normalize(rand(k, k))
    return pi, O, T

def generate_marginals(pi, O, T):
    Z1 = pi
    M1 = O.dot(Z1.T) # d x 1
    Z12 = T.dot(np.diag(pi))
    M12 = O.dot(Z12).dot(O.T) # d x d

    return Z1, M1, Z12, M12

def recover_pseudoinverse(M1, M12, O):
    Oi = pinv(O)
    Z1_ = Oi.dot(M1)
    Z12_ = Oi.dot(M12).dot(Oi.T)

    return Z1_, Z12_

def recover_pi(data, O, iters = 100, eps = 1e-5):
    d, k = O.shape
    # Initialize pi randomly
    lhood_ = -np.inf
    pi = normalize(rand(k))

    for i in xrange(iters):
        pi_ = zeros(k)
        lhood = 0
        for (xs, prob) in data.items():
            x1, _, _ = xs
            z = pi * O[x1]
            lhood += prob * np.log(sum(z))
            pi_ += prob * (z / z.sum()) 
        assert np.allclose(pi_.sum(), 1.)

        assert lhood - lhood_ > -1e-5
        lhood_ = lhood

        done = np.allclose(pi, pi_, eps)
        pi = pi_
        if done:
            break

    return pi

def recover_Z12(data, O, iters = 100, eps = 1e-5, Z12 = None):
    d, k = O.shape
    # Initialize pi randomly
    lhood_ = -np.inf
    if Z12 is None:
        Z12 = normalize(rand(k,k))


    for i in xrange(iters):
        Z12_ = zeros((k, k))
        lhood = 0
        for (xs, prob) in data.items():
            x1, x2, _ = xs
            z = Z12 * np.outer(O[x2], O[x1])
            lhood += prob * np.log(z.sum())
            Z12_ += prob * (z / z.sum())
        assert np.allclose(Z12_.sum(), 1.)

        assert lhood - lhood_ > -1e-5

        done = np.allclose(Z12, Z12_, eps)

        #ipdb.set_trace()
        Z12 = Z12_
        lhood_ = lhood
        if done:
            break

    return Z12

def generate_data(pi, O, T):
    """
    Generates data fragments of length 3
    """
    d, k = O.shape
    L = 3
    data = Counter()
    for hidden in it.product(*it.repeat(xrange(k), L)):
        for obs in it.product(*it.repeat(xrange(d), L)):
            h1, h2, h3 = hidden
            x1, x2, x3 = obs
            prob = pi[h1] * O[x1,h1] * T[h2,h1] * O[x2,h2] * T[h3,h2] * O[x3,h3]
            data[obs] += prob
    
    assert np.allclose(sum(data.values()), 1, 1e-4)

    return data

def recover_composite_likelihood(data, O, iters = 1000, eps = 1e-7):
    Z1_ = recover_pi(data, O, iters, eps)
    Z12_ = recover_Z12(data, O, iters, eps)

    return Z1_, Z12_

def normalize(obj):
    return obj / obj.sum()

def column_normalize(obj):
    if len(obj.shape) == 2:
        m, n = obj.shape
        return obj / np.repeat( np.atleast_2d(obj.sum(0)), m, axis=0 )
    else:
        raise AttributeError("Incorrect shape")

def noise_data(data, eps):
    for key, val in data.items():
        data[key] = abs(val + eps * randn())
    z = sum(data.values()) 
    for key, val in data.items():
        data[key] /= z
    assert np.allclose(sum(data.values()), 1.)

    return data

def project_data(data):
    d = len(data.keys()[0])
    M1 = zeros(d)
    M12 = zeros((d,d))
    for key, val in data.items():
        x1, x2, x3 = key
        M1[x1] += val
        M12[x1,x2] += val
    return M1, M12

def do_experiment(params, data, args):
    pi, O, T = params
    noise, = args.noise, 
    eps, iters = args.eps, args.iters 

    data = noise_data(data, noise)
    M1_, M12_  = project_data(data)

    Z1_pi, Z12_pi = recover_pseudoinverse(M1_, M12_, O)
    Z1_cl, Z12_cl = recover_composite_likelihood(data, O, iters=iters, eps=eps)

    return [Z1_pi, Z12_pi], [Z1_cl, Z12_cl]


def do_command(args):
    np.random.seed(args.seed)

    K, D = args.k, args.d
    noise, attempts = args.noise, args.attempts
    eps, iters = args.eps, args.iters 

    pi, O, T = generate_random_parameters(K, D)
    Z1, M1, Z12, M12 = generate_marginals(pi, O, T)

    data = generate_data(pi, O, T)

    dZ_pi, dZ_cl = [], []

    for attempt in xrange(attempts):
        # Add noise
        (Z1_pi, Z12_pi), (Z1_cl, Z12_cl) = do_experiment([pi, O, T], data, args)
        e_pi = norm(Z1_pi - Z1) + norm(Z12_pi - Z12)
        e_cl = norm(Z1_cl - Z1) + norm(Z12_cl - Z12)
        dZ_pi.append(e_pi)
        dZ_cl.append(e_cl)
        if args.debug:
            print attempt, e_pi, e_cl

    dZ_pi, dZ_cl = np.array(dZ_pi), np.array(dZ_cl)

    print "noise=%f\tdZ_pi=%f\tstd_dZ_pi=%f\tdZ_cl=%f\tstd_dZ_cl=%f"%( noise, dZ_pi.mean(), dZ_pi.std(), dZ_cl.mean(), dZ_cl.std() )

    return dZ_pi, dZ_cl

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='Pseudo-inversion magic' )
    parser.add_argument( '--k', type=int, default=2, help="number of clusters" )
    parser.add_argument( '--d', type=int, default=3, help="number of dimensions" )
    parser.add_argument( '--noise', type=float, default=1e-3, help="noise added" )
    parser.add_argument( '--eps', type=float, default=1e-9, help="precision of EM" )
    parser.add_argument( '--iters', type=int, default=10000, help="precision of EM" )
    parser.add_argument( '--attempts', type=int, default=1000, help="number of attempts" )
    parser.add_argument( '--debug', type=bool, default=False, help="print iteration output" )
    parser.add_argument( '--seed', type=int, default=12, help="random-seed" )
    #parser.add_argument( 'method', choices=['spectral','piecewise'], default='spectral', )
    parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)
