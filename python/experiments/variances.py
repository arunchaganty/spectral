#!/usr/bin/env python2.7
"""
A simple script to compare composite likelihood and pseudo-inversion based estimators.
"""

import ipdb

import numpy as np
from numpy.linalg import inv, svd, pinv, norm, eigvals
from numpy.random import rand, randn
from numpy import zeros, ones, eye, trace
import itertools as it
import sys

from collections import Counter

np.random.seed(12)

def generate_random_instance(k, d, badO = False):
    z = normalize(rand(k))
    if badO:
        O = 0.9 * column_normalize(rand(d-1, k))
        O = np.vstack((O, 0.1 * np.ones(k)))
    else:
        O = column_normalize(rand(d, k))
    assert np.allclose(O.sum(), k, 1e-4)
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

def recover_composite_likelihood(O, mu, iters = 1000, eps = 1e-4, z0 = None):
    d, k = O.shape
    # Initialize pi randomly
    lhood_ = -np.inf
    if z0 is None or (z0 < 0).any() or (z0 > 1).any():
        print "random init"
        z = normalize(rand(k))
    else:
        z = np.hstack((z0, 1 - z0.sum()))

    #ipdb.set_trace()

    for i in xrange(iters):
        z_ = zeros(k)
        lhood = 0
        for x, prob in enumerate(mu):
            z_marg = O[x] * z
            lhood += prob * np.log(z_marg.sum())
            z_ += prob * (z_marg / z_marg.sum())
        assert np.allclose(z_.sum(), 1., 1e-4)

        assert lhood - lhood_ > -1e-5
        lhood_ = lhood

        done = norm(z - z_) < eps
        z = z_
        if done:
            break

    return z[:-1]

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
    zt_cl = recover_composite_likelihood(O, mu_, z0 = zt_pi)

    return mut_, zt_pi, zt_cl


def do_command(args):
    np.random.seed(args.seed)

    K, D = args.k, args.d
    attempts = int(args.attempts)
    eps, iters = args.eps, args.iters 

    z, O, mu = generate_random_instance(K, D, args.badO)
    mut = mu[:-1]
    zt = z[:-1]
    Ondk = O[:-1, -1]
    Ot = O[:-1, :-1] - Ondk.reshape(D-1, 1).dot( ones((1,K-1)) )
    print "ones residual", norm( ones((D-1,1)).T.dot(Ot) )

    dZ_pi, dZ_cl = [], []

    for attempt in xrange(attempts):
        mut_, zt_pi, zt_cl = do_experiment(O, Ot, Ondk, mu, args)
        if attempt % attempts/20 == 0:
            sys.stderr.write('.')
            sys.stderr.flush()

        e_pi = zt - zt_pi
        e_cl = zt - zt_cl
        dZ_pi.append(e_pi)
        dZ_cl.append(e_cl)
        if args.debug:
            print attempt, norm(e_pi), norm(e_cl)
    dZ_pi, dZ_cl = np.array(dZ_pi), np.array(dZ_cl)
    sys.stderr.write('\n')

    Oti = pinv(Ot)
    P = Ot.dot(Oti)
    Dt = np.diag(mut)
    mut = np.atleast_2d(mut)
    o = ones((D-1,1))
    v = P.dot(o)
    # Theory!
    S_pi = Oti.dot(Dt - Dt.dot(o).dot(o.T).dot(Dt)).dot(Oti.T) / args.n
    S_cl = Oti.dot(Dt - Dt.dot(v).dot(v.T).dot(Dt)/(1 - o.T.dot(Dt).dot(o) + v.T.dot(Dt).dot(v))).dot(Oti.T) / args.n
    
    # Practice!
    S_pi_ = (dZ_pi - dZ_pi.mean()).T.dot(dZ_pi - dZ_pi.mean()) / attempts
    S_cl_ = (dZ_cl - dZ_cl.mean()).T.dot(dZ_cl - dZ_cl.mean()) / attempts
    sys.stderr.write( "S_pi*" + str(S_pi ) + "\n")
    sys.stderr.write( "S_pi^" + str(S_pi_) + "\n")
    sys.stderr.write( "S_cl*" + str(S_cl ) + "\n")
    sys.stderr.write( "S_cl^" + str(S_cl_) + "\n")

    sys.stderr.write( "practice l" + str( eigvals(S_pi_ - S_cl_) ) + "\n")
    sys.stderr.write( "theory l" + str( eigvals(S_pi - S_cl) ) + "\n")

    sys.stderr.write( "practice tr" + str( trace(S_pi_ - S_cl_) ) + "\n" )
    sys.stderr.write( "theory tr" + str( trace(S_pi - S_cl) ) + "\n" )

    e_pi, e_cl = np.sqrt((dZ_pi**2).sum(1)), np.sqrt((dZ_cl**2).sum(1))

    print "n=%f\tdZ_pi=%f\tstd_dZ_pi=%f\tdZ_cl=%f\tstd_dZ_cl=%f"%( args.n, e_pi.mean(), e_pi.std(), e_cl.mean(), e_cl.std() )

    return dZ_pi, dZ_cl

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='Pseudo-inversion magic' )
    parser.add_argument( '--k', type=int, default=2, help="number of clusters" )
    parser.add_argument( '--d', type=int, default=3, help="number of dimensions" )
    parser.add_argument( '--n', type=float, default=1e3, help="noise added" )
    parser.add_argument( '--eps', type=float, default=1e-9, help="precision of EM" )
    parser.add_argument( '--iters', type=int, default=10000, help="precision of EM" )
    parser.add_argument( '--attempts', type=float, default=1000, help="number of attempts" )
    parser.add_argument( '--badO', type=bool, default=False, help="Use an O that is bad" )
    parser.add_argument( '--debug', type=bool, default=False, help="print iteration output" )
    parser.add_argument( '--seed', type=int, default=12, help="random-seed" )
    #parser.add_argument( 'method', choices=['spectral','piecewise'], default='spectral', )
    parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)
