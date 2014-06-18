#!/usr/bin/env python2.7
"""
This experiment seeks to produce a plot of likelihood and error in
expected moments (countsErr) as a function of samples (n), comparing
different partial measurements percentages.
"""

import os
import random
import numpy as np
import pandas as pd
import scabby

EXPT_NAME = "hmm_window"

KD_VALUES = [(3,3),] # (2,3), (3,3), (3,5),]# (3,10), (5,10)]
WINDOWS_SIZES = [1,]

def get_settings(args):
    for k,d in KD_VALUES:
        l = 5
        assert k <= d
        for w in WINDOWS_SIZES:
            for params_seed in xrange( args.nParams ):
                yield dict(locals())

def do_run(args):
    # Create directory
    if not os.path.exists(args.exptdir):
        os.mkdir(args.exptdir)
    if not os.path.exists(args.execdir):
        os.mkdir(args.execdir)

    cmd = '\
./run.sh learning.experiments.SpectralMeasurements\
 -execPoolDir {args.execdir}\
 -modelType hmm\
 -K {k} -D {d} -L {l} -windowSize {w}\
 -trueParamsRandom {params_seed}\
 -initParamsNoise 1.0\
 -mode SpectralMeasurements\
 -betaRegularization 1e-2\
 -initializeWithExact\
 -smoothMeasurements 1e-5\
 -SpectralMeasurements.MeasurementsEM.iters 200 -eIters 1000'

    settings = get_settings(args)

    if args.parallel:
        scabby.parallel_spawn( args.exptdir, "qstart-short", cmd, args.njobs, settings )
    else:
        for setting in settings:
            scabby.safe_run(cmd.format(**setting))

def do_process(args):
    import plumbum as pb

    for k, d in KD_VALUES:
        print k, d
        cmd = 'tab.py extract \
 --execdir {args.execdir} \
 --filters K={k} D={d} modelType=hmm\
 --keys preconditioning genNumExamples paramsError countsError fit-perp'.format(**locals())
        raw_path = os.path.join( args.exptdir, 'hmm-{k}-{d}-{preconditioning}.raw.tab'.format(**locals()) )
        (pb.local['python2.7'][cmd.split()] > raw_path)()

        agg_path = os.path.join( args.exptdir, 'hmm-{k}-{d}-{preconditioning}.agg.tab'.format(**locals()) )
        if args.best:
            cmd = 'tab.py agg --mode min genNumExamples preconditioning'.format(**locals())
        else:
            cmd = 'tab.py agg genNumExamples preconditioning'.format(**locals())
        cmd_ = 'tab.py sort genNumExamples'.format(**locals())
        (pb.local['cat'][raw_path] | pb.local['python2.7'][cmd.split()] | pb.local['python2.7'][cmd_.split()] > agg_path)()

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
    run_parser.add_argument( '--nParams', type=int, default=5, help="# of parameters to use" )
    #run_parser.add_argument( 'extra-args', type=str, nargs='+', help="Additional arguments for the actual program" )
    run_parser.set_defaults(func=do_run)

    plot_parser = subparsers.add_parser('process', help='Plot results from the experiment' )
    plot_parser.add_argument( '--best', action="store_true", help="When plotting, choose the best over the different runs." )
    plot_parser.set_defaults(func=do_process)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)


