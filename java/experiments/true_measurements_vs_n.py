#!/usr/bin/env python2.7
"""
This experiment seeks to produce a plot of likelihood and error in
expected moments (countsErr) as a function of samples (n), comparing
different partial measurements percentages.
"""

import plumbum as pb
import os
from subprocess import Popen
import shlex
import random

EXPT_NAME = "true_measurements_vs_n"

KD_VALUES = [(2,2), (2,3), (3,3), (3,5), (3,10), (5,10)]
MEASUREMENT_PROB_VALUES = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
N_VALUES = [100, 500, 1000, 5000, 10000, 20000]

def do_run(args):
    #random.seed(args.seed)
    #initial_seed = random.randint(0,255)

    # Create directory
    if not os.path.exists(args.execdir):
        os.mkdir(args.execdir)

    # Run for different k,d configurations
    for k,d in KD_VALUES:
        assert k <= d

        for measurement_prob in MEASUREMENT_PROB_VALUES:
            for n in N_VALUES:
                # Run some number of iterations
                for initialization_seed in xrange( args.runAverageIters ):
                    cmd = '\
./run.sh learning.models.loglinear.SpectralMeasurements\
 -execPoolDir {args.execdir}\
 -model {args.model}\
 -K {k} -D {d} -L 3\
 -initRandom {initialization_seed}\
 -measurementProb {measurement_prob}\
 -genNumExamples {n}\
 -SpectralMeasurements.MeasurementsEM.iters 50 -eIters 500'.format(**locals())

                    # Actually run the script
                    proc = Popen(shlex.split(cmd))
                    proc.wait()

def do_plot(args):
    import scabby
    # Load a all the files that meet the settings and average over them
    # to generate a couple of data points
    
    pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='Study the effect of partial measurements on the recovery accuracy.' )
    parser.add_argument( '--seed', type=int, default=23, help="A seed to generate seeds!" )
    parser.add_argument( '--execdir', type=str, default='state/execs/%s/'%(EXPT_NAME,), help="Location of the exec directories" )
    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser('run', help='Run the experiment' )
    run_parser.add_argument( '--runAverageIters', type=int, default=5, help="Number of different initial seeds to run with" )
    run_parser.add_argument( '--model', type=str, default="mixture", choices=["mixture","hmm"], help="Model to use" )
    #run_parser.add_argument( 'extra-args', type=str, nargs='+', help="Additional arguments for the actual program" )
    run_parser.set_defaults(func=do_run)

    plot_parser = subparsers.add_parser('plot', help='Plot results from the experiment' )
    plot_parser.set_defaults(func=do_plot)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)
