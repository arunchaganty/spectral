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


EXPT_NAME = "SpectralMeasurements"

KD_VALUES = [(2,2), (2,3),(3,3), (3,5), (3,10), (5,10)]
N_VALUES = [1000, 2000, 5000, 7000, 10000, 20000, 50000, 70000, 1000000, 200000, 500000, 700000]
PRECONDITIONG_VALUES = [0.0, 1e-3] #1e-2, 1e-3]

def get_settings(args):
    for k,d in KD_VALUES:
        assert k <= d
        for n in N_VALUES:
            for preconditioning in PRECONDITIONG_VALUES:
                for model_seed in xrange( args.instantiations ):
                    for initialization_seed in xrange( args.initializations ):
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
 -K {k} -D {d} -L 4\
 -initRandom {initialization_seed}\
 -initParamsNoise 1.0\
 -trueParamsRandom {model_seed}\
 -mode SpectralMeasurements\
 -preconditioning {preconditioning}\
 -betaRegularization 1e-2\
 -genNumExamples {n}\
 -smoothMeasurements 1e-5\
 -SpectralMeasurements.MeasurementsEM.iters 200 -eIters 1000'

    settings = get_settings(args)

    if args.parallel:
        scabby.parallel_spawn( args.exptdir, "qstart", cmd, args.njobs, settings )
    else:
        for setting in settings:
            scabby.safe_run(cmd.format(**setting))

def do_process(args):
    import plumbum as pb

    for k, d in KD_VALUES:
        print k, d
        cmd = 'tab.py extract \
 --execdir {args.execdir} \
 --filters K={k} D={d} modelType={args.model} \
 --keys trueParamsRandom genNumExamples preconditioning paramsError marginalError countsError fit-perp'.format(**locals())
        raw_path = os.path.join( args.exptdir, '{args.model}-{k}-{d}.raw.tab'.format(**locals()) )
        (pb.local['python2.7'][cmd.split()] > raw_path)()

        agg_path = os.path.join( args.exptdir, '{args.model}-{k}-{d}.agg.tab'.format(**locals()) )
        if args.best:
            cmd = 'tab.py agg --mode min genNumExamples trueParamsRandom preconditioning'.format(**locals())
        else:
            cmd = 'tab.py agg genNumExamples trueParamsRandom preconditioning'.format(**locals())
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
    run_parser.add_argument( '--initializations', type=int, default=3, help="Number of different initial seeds to run with" )
    run_parser.add_argument( '--instantiations', type=int, default=3, help="Number of different initial seeds to run with" )
    run_parser.add_argument( '--model', type=str, default="mixture", choices=["mixture","hmm","grid"], help="Model to use" )
    #run_parser.add_argument( 'extra-args', type=str, nargs='+', help="Additional arguments for the actual program" )
    run_parser.set_defaults(func=do_run)

    plot_parser = subparsers.add_parser('process', help='Plot results from the experiment' )
    plot_parser.add_argument( '--model', type=str, default="mixture", choices=["mixture","hmm","grid"], help="Model to use" )
    plot_parser.add_argument( '--best', action="store_true", help="When plotting, choose the best over the different runs." )
    plot_parser.set_defaults(func=do_process)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)

