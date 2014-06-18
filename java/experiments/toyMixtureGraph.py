#!/usr/bin/env python2.7
"""
This experiment seeks to produce a plot of likelihood and error in
expected moments (countsErr) as a function of samples (n), comparing
different partial measurements percentages and spectral.
"""

import os
import random
import numpy as np
import scabby
import itertools as it

EXPT_NAME = "ToyMixtureGraph"
MODEL = "mixture"

KD_VALUES = [(2,2), (2,3), (3,3), (3,5),]# (3,10), (5,10)]
N_VALUES = [1e3, 2e3, 5e3, 7e3,
            1e4, 2e4, 5e4, 7e4,
            1e5, 2e5, 5e5, 7e5,
            5e7 # equivalent to infinity
            ]

MEASUREMENT_PROB_VALUES = [1.0, 0.7, 0.3, 0.0]
NOISE_VALUES = [0.,] # 1e-1,] #1e-2]

PRECONDITIONG_VALUES = [0.0,]# 1e-3] #1e-2, 1e-3]

def get_true_settings(args):
    model = MODEL
    preconditioning = 0.0
    mode = "TrueMeasurements"
    for k,d in KD_VALUES:
        assert k <= d
        for measured_fraction in MEASUREMENT_PROB_VALUES:
            for n in N_VALUES:
                for measurement_noise in NOISE_VALUES:
                    for model_seed in xrange( args.instantiations ):
                        for initialization_seed in xrange( args.initializations ):
                            yield dict(locals())

def get_spectral_settings(args):
    model = MODEL
    measured_fraction = 1.0
    measurement_noise = 0.0
    mode = "SpectralMeasurements"
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
./run.sh learning.experiments.SpectralMeasurements\
 -execPoolDir {args.execdir}\
 -modelType {model}\
 -K {k} -D {d} -L 3\
 -initRandom {initialization_seed}\
 -initParamsNoise 1.0\
 -mode {mode}\
 -trueParamsRandom {model_seed}\
 -measuredFraction {measured_fraction}\
 -trueMeasurementNoise {measurement_noise}\
 -preconditioning {preconditioning}\
 -genNumExamples {n}\
 -SpectralMeasurements.MeasurementsEM.iters 200 -eIters 10000'

    settings = it.chain(get_true_settings(args),get_spectral_settings(args))
    if args.pretend:
        count = len(list(settings))
        print "%d settings split over %d runs (%d each)" %(count, args.njobs, count/args.njobs+1)
    elif args.parallel:
        scabby.parallel_spawn( args.exptdir, "qstart", cmd, args.njobs, settings )
    else:
        for setting in settings:
            scabby.safe_run(cmd.format(**setting))

def do_process(args):
    import plumbum as pb
    model = MODEL

    for k, d in KD_VALUES:
        for mode in ["TrueMeasurements", "SpectralMeasurements"]:
            print k, d, mode
            cmd = 'tab.py extract\
     --execdir {args.execdir}\
     --filters K={k} D={d} modelType={model} mode={mode}\
     --keys trueParamsRandom genNumExamples measuredFraction trueMeasurementNoise paramsError countsError marginalError fit-perp'.format(**locals())
            raw_path = os.path.join( args.exptdir, '{model}-{k}-{d}-{mode}.raw.tab'.format(**locals()) )

            print (pb.local['python2.7'][cmd.split()] > raw_path)
            (pb.local['python2.7'][cmd.split()] > raw_path)()

            agg1_path = os.path.join( args.exptdir, '{model}-{k}-{d}-{mode}.agg1.tab'.format(**locals()) )
            if args.best:
                cmd = 'tab.py agg --mode min genNumExamples trueParamsRandom measuredFraction trueMeasurementNoise'.format(**locals())
            else:
                cmd = 'tab.py agg genNumExamples trueParamsRandom measuredFraction trueMeasurementNoise'.format(**locals())

            print (pb.local['cat'][raw_path] | pb.local['python2.7'][cmd.split()] > agg1_path)
            (pb.local['cat'][raw_path] | pb.local['python2.7'][cmd.split()] > agg1_path)()

            # Micro-average
            if args.best:
                agg_path = os.path.join( args.exptdir, '{model}-{k}-{d}-{mode}.best.tab'.format(**locals()) )
            else:
                agg_path = os.path.join( args.exptdir, '{model}-{k}-{d}-{mode}.agg.tab'.format(**locals()) )
            cmd_agg = 'tab.py agg genNumExamples measuredFraction trueMeasurementNoise'.format(**locals()) 
            cmd_sort = 'tab.py sort trueMeasurementNoise measuredFraction genNumExamples'.format(**locals())

            print (pb.local['cat'][agg1_path] | pb.local['python2.7'][cmd_agg.split()] | pb.local['python2.7'][cmd_sort.split()] > agg_path)
            (pb.local['cat'][agg1_path] | pb.local['python2.7'][cmd_agg.split()] | pb.local['python2.7'][cmd_sort.split()] > agg_path)()





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='Study the effect of partial measurements on the recovery accuracy.' )
    parser.add_argument( '--seed', type=int, default=23, help="A seed to generate seeds!" )
    parser.add_argument( '--execdir', type=str, default='state/execs/%s/'%(EXPT_NAME,), help="Location of the exec directories" )
    parser.add_argument( '--exptdir', type=str, default='state/expts/%s/'%(EXPT_NAME,), help="Location of the exec directories" )
    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser('run', help='Run the experiment' )
    run_parser.add_argument( '--pretend', action='store_true', help="Report how many settings you'd have" )
    run_parser.add_argument( '--parallel', action='store_true', help="Spawn parallel jobs?" )
    run_parser.add_argument( '--njobs', type=int, default=10, help="How many parallel jobs?" )
    run_parser.add_argument( '--initializations', type=int, default=5, help="Number of different initial seeds to run with" )
    run_parser.add_argument( '--instantiations', type=int, default=5, help="Number of different initial seeds to run with" )
    #run_parser.add_argument( '--model', type=str, default="mixture", choices=["mixture","hmm", "grid"], help="Model to use" )
    #run_parser.add_argument( 'extra-args', type=str, nargs='+', help="Additional arguments for the actual program" )
    run_parser.set_defaults(func=do_run)

    plot_parser = subparsers.add_parser('process', help='Plot results from the experiment' )
    #plot_parser.add_argument( '--model', type=str, default="mixture", choices=["mixture","hmm","grid"], help="Model to use" )
    plot_parser.add_argument( '--best', action="store_true", help="When plotting, choose the best over the different runs." )
    plot_parser.set_defaults(func=do_process)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)

