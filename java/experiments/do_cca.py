#!/usr/bin/env python2.7
"""
Plot something with CCA
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import inv, norm

def do_command(args):
    data = np.loadtxt(args.data)
    X = data[:,:-1]
    y = data[:,-1]
    n, d = X.shape

    # Mean average
    X = X - X.mean(0)
    y_0 = y - y.mean(0)

    if args.mode == "cca":
        # Compute covariances
        S_xx = X.T.dot(X) / (n-1)
        S_xy = X.T.dot(y_0) / (n-1)

        # C is proportional to S_xx^{-1} S_xy
        c = inv(S_xx).dot(S_xy)
        c = c / norm(c)
    elif args.mode == "random":
        c = np.random.rand(d)
    else:
        raise AttributeError("Invalid mode ", args.mode)

    # Now plot
    data = np.vstack( (X.dot(c), y) ).T
    r_data = data[ data.T[0].argsort(), : ]

    true = data[:args.true_rows]
    em = data[args.true_rows:args.true_rows+args.em_rows,:]

    plt.xlabel("Projected coordinates")
    plt.ylabel("Log-likelihood")
    plt.plot(r_data.T[0], r_data.T[1])
    plt.plot(true.T[0], true.T[1], 'g*', markersize=30)
    plt.plot(em.T[0], em.T[1], 'r.', markersize=15)

    if args.output is not None and len(args.output) > 0:
        plt.savefig(args.output)
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='Plot EM using CCA' )
    parser.add_argument( '--mode', choices=["cca","random"], help="mode of projection" )
    parser.add_argument( '--true-rows', type=int, default=1, help="number of true param rows" )
    parser.add_argument( '--em-rows', type=int, default=100, help="number of EM rows" )
    parser.add_argument( '--output', type=str, help="Where to save" )
    parser.add_argument( 'data', type=str, help="Path to file" )
    parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)
