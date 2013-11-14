#!/usr/bin/env python2.7
"""
This experiment seeks to produce a plot of likelihood and error in
expected moments (countsErr) as a function of samples (n), comparing
different partial measurements percentages.
"""

import os
from subprocess import Popen
import shlex
import random
import numpy as np
import scabby

EXPT_NAME = "true_measurements_vs_n"

KD_VALUES = [(2,2), (2,3), (3,3),]# (3,5), (3,10), (5,10)]
MEASUREMENT_PROB_VALUES = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
N_VALUES = [100, 200, 500, 700, 1000, 2000, 5000, 7000, 10000, 20000, 50000, 70000, 100000]

def get_settings(args):
    for k,d in KD_VALUES:
        assert k <= d
        for measurement_prob in MEASUREMENT_PROB_VALUES:
            for n in N_VALUES:
                for initialization_seed in xrange( args.repeatIters ):
                    yield dict(locals())

def do_run(args):
    #random.seed(args.seed)
    #initial_seed = random.randint(0,255)

    # Create directory
    if not os.path.exists(args.exptdir):
        os.mkdir(args.exptdir)
    if not os.path.exists(args.execdir):
        os.mkdir(args.execdir)

    cmd = '\
./run.sh learning.models.loglinear.SpectralMeasurements\
 -execPoolDir {args.execdir}\
 -modelType {args.model}\
 -K {k} -D {d} -L 3\
 -initRandom {initialization_seed}\
 -measurementProb {measurement_prob}\
 -genNumExamples {n}\
 -SpectralMeasurements.MeasurementsEM.iters 50 -eIters 500'

    settings = get_settings(args)

    if args.parallel:
        scabby.parallel_spawn( args.exptdir, "qstart-short", cmd, args.njobs, settings )
    else:
        for setting in settings:
            scabby.safe_run(cmd.format(**setting))

def do_plot(args):
    import scabby
    #import matplotlib.pyplot as plt 
    # Load a all the files that meet the settings and average over them
    # to generate a couple of data points
    YS = ['countsError', 'paramsError', 'fit-perp', 'true-perp']

    def get_measurement_prob(exec_dir):
        for line in open(os.path.join(exec_dir, 'log')):
            line = line.strip()
            if line.startswith('sum_counts:'):
                _, prob = line.split(':')
                return float(prob)

    # One plot per k,d value
    for k,d in KD_VALUES:
        assert k <= d
        print k,d

        plot = []
        # One line per measurement_prob
        for measurement_prob in MEASUREMENT_PROB_VALUES:
            print measurement_prob
            true_measurement_prob = measurement_prob
            line = []
            # One data point per n#
            for n in N_VALUES:
                #avg, cnt = [0.0,0.0,0.0,0.0], 0
                best, cnt = [float('inf'), float('inf'), float('-inf'), float('-inf'), ], 0
                for exec_dir in  scabby.get_execs( args.execdir, 
                        K=k, D=d, modelType=args.model,
                        measurementProb=measurement_prob,
                        genNumExamples=n ):
                    out = scabby.read_options(os.path.join( exec_dir, 'output.map' ))
                    try:
                        x = map( lambda key: float(out[key]), YS )
                        #avg, cnt = scabby.running_average( avg, cnt, x )
                        best[0] = min(best[0], x[0])
                        best[1] = min(best[1], x[1])
                        best[2] = max(best[2], x[2])
                        best[3] = max(best[3], x[3])
                        cnt = cnt + 1
                        true_measurement_prob = get_measurement_prob(exec_dir)
                    except KeyError:
                        continue
                if cnt > 0:
                    #line.append( [n] + avg )
                    line.append( [n] + best )
            line = np.array(line)
            print true_measurement_prob, line
            plot.append( ("%0.2f"%(true_measurement_prob,), line) )
        figs = scabby.plot_many( plot, xlabel='n', ylabels=YS )
        for fig, y in zip(figs, YS):
            fig.savefig(os.path.join(args.exptdir, '{args.model}-min-{y}-{k}-{d}.png'.format(**locals())))
        #print plot

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='Study the effect of partial measurements on the recovery accuracy.' )
    parser.add_argument( '--seed', type=int, default=23, help="A seed to generate seeds!" )
    parser.add_argument( '--execdir', type=str, default='state/execs/%s/'%(EXPT_NAME,), help="Location of the exec directories" )
    parser.add_argument( '--exptdir', type=str, default='state/expts/%s/'%(EXPT_NAME,), help="Location of the exec directories" )
    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser('run', help='Run the experiment' )
    run_parser.add_argument( '--parallel', action='store_true', help="Spawn parallel jobs?" )
    run_parser.add_argument( '--njobs', type=int, default=10, help="How many parallel jobs?" )
    run_parser.add_argument( '--repeatIters', type=int, default=5, help="Number of different initial seeds to run with" )
    run_parser.add_argument( '--model', type=str, default="mixture", choices=["mixture","hmm"], help="Model to use" )
    #run_parser.add_argument( 'extra-args', type=str, nargs='+', help="Additional arguments for the actual program" )
    run_parser.set_defaults(func=do_run)

    plot_parser = subparsers.add_parser('plot', help='Plot results from the experiment' )
    plot_parser.add_argument( '--model', type=str, default="mixture", choices=["mixture","hmm"], help="Model to use" )
    plot_parser.set_defaults(func=do_plot)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)
