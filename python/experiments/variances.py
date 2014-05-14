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

def generate_random_instance(k, d):
    z = normalize(rand(k))
    O = column_normalize(rand(d, k))
    mu = O.dot(z)
    return z, O, mu

def generate_marginals(pi, O, T):
    Z1 = pi
    M1 = O.dot(Z1.T) # d x 1
    Z12 = T.dot(np.diag(pi))
    M12 = O.dot(Z12).dot(O.T) # d x d

    return Z1, M1, Z12, M12

def recover_pseudoinverse(Ot, Ondk, mut_):
    Oti = pinv(Ot)
    zt_ = Oti.dot(mut_ - Ondk)

    return zt_

def recover_composite_likelihood(O, mu, iters = 100, eps = 1e-5):
    d, k = O.shape
    # Initialize pi randomly
    lhood_ = -np.inf
    zt = normalize(rand(k))[:-1]

    #ipdb.set_trace()

    for i in xrange(iters):
        zt_ = zeros(k - 1)
        lhood = 0
        for x, prob in enumerate(mu):
            z_marg = np.hstack( (O[x,:-1] * zt, O[x,-1] * (1 - zt).sum()) )
            lhood += prob * np.log(z_marg.sum())
            zt_ += prob * (z_marg / z_marg.sum())[:-1]

        assert zt_.sum() < 1.

        assert lhood - lhood_ > -1e-5
        lhood_ = lhood

        done = np.allclose(zt, zt_, eps)
        zt = zt_
        if done:
            break

    return zt

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

def do_experiment(O, Ot, Ondk, mu, args):
    # Generate mu_hat
    mu_ = np.random.multinomial(int(args.n), mu) / args.n

    mut_ = mu_[:-1]
    zt_pi = recover_pseudoinverse(Ot, Ondk, mut_)
    zt_cl = recover_composite_likelihood(O, mu_)

    return mut_, zt_pi, zt_cl


def do_command(args):
    np.random.seed(args.seed)

    K, D = args.k, args.d
    attempts = args.attempts
    eps, iters = args.eps, args.iters 

    z, O, mu = generate_random_instance(K, D)
    mut = mu[:-1]
    zt = z[:-1]
    Ondk = O[:-1, -1]
    Ot = O[:-1, :-1] - Ondk.reshape(D-1, 1).dot( ones((1,K-1)) )

    dZ_pi, dZ_cl = [], []

    for attempt in xrange(attempts):
        mut_, zt_pi, zt_cl = do_experiment(O, Ot, Ondk, mu, args)

        e_pi = norm(zt - zt_pi)
        e_cl = norm(zt - zt_cl)
        dZ_pi.append(e_pi)
        dZ_cl.append(e_cl)
        if args.debug:
            print attempt, e_pi, e_cl

    dZ_pi, dZ_cl = np.array(dZ_pi), np.array(dZ_cl)

    print "n=%f\tdZ_pi=%f\tstd_dZ_pi=%f\tdZ_cl=%f\tstd_dZ_cl=%f"%( args.n, dZ_pi.mean(), dZ_pi.std(), dZ_cl.mean(), dZ_cl.std() )

    return dZ_pi, dZ_cl

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='Pseudo-inversion magic' )
    parser.add_argument( '--k', type=int, default=2, help="number of clusters" )
    parser.add_argument( '--d', type=int, default=3, help="number of dimensions" )
    parser.add_argument( '--n', type=float, default=1e3, help="noise added" )
    parser.add_argument( '--eps', type=float, default=1e-9, help="precision of EM" )
    parser.add_argument( '--iters', type=int, default=10000, help="precision of EM" )
    parser.add_argument( '--attempts', type=int, default=1000, help="number of attempts" )
    parser.add_argument( '--debug', type=bool, default=False, help="print iteration output" )
    parser.add_argument( '--seed', type=int, default=12, help="random-seed" )
    #parser.add_argument( 'method', choices=['spectral','piecewise'], default='spectral', )
    parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)
