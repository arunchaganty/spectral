#!/usr/bin/env python2.7
"""
Plot Jacobian singular value vs. error.
"""

import os
import random
import numpy as np
import scabby

EXPT_NAME = "PairwiseHMMRecovery"

KD_VALUES = [(2,2), (2,3),] # (3,3),]# (3,5), (3,10), (5,10)]
L = [3, 5]
#MODES = ["SpectralConvex", "SpectralInitialization"]
MODES = ["SpectralConvex", "SpectralInitialization"]

def get_settings(args):
    for mode in MODES:
        for k,d in KD_VALUES:
            for l in L:
                for initO in ["true", "false"]:
                    assert k <= d
                    for model_seed in xrange( args.instantiations ):
                        for init_seed in xrange( args.initializations ):
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
./run.sh learning.experiments.PairwiseHMMRecovery\
 -execPoolDir {args.execdir}\
 -mode {mode} -N 1e5 \
 -stateCount {k} -emissionCount {d} -L {l}\
 -initExactO {initO}\
 -paramRandom {model_seed}\
 -initRandom {init_seed}'

    settings = get_settings(args)

    if args.pretend:
        count = len(list(settings))
        print "%d settings split over %d runs (%d each)" %(count, args.njobs, count/args.njobs+1)
    elif args.parallel:
        scabby.parallel_spawn( args.exptdir, "qstart-short", cmd, args.njobs, settings )
    else:
        for setting in settings:
            scabby.safe_run(cmd.format(**setting))

def do_process(args):
    import plumbum as pb

    for k, d in KD_VALUES:
        for mode in MODES:
            print k, d
            cmd = 'tab.py extract \
     --execdir {args.execdir} \
     --filters stateCount={k} emissionCount={d} mode={mode}\
     --keys initExactO l O-sigmak O-K full-sigmak full-K piecewise-sigmak piecewise-K O-error T-error pi-error initial-paramsError'.format(**locals())
            raw_path = os.path.join( args.exptdir, '{args.model}-{k}-{d}-{mode}.raw.tab'.format(**locals()) )
            print (pb.local['python2.7'][cmd.split()] > raw_path)
            (pb.local['python2.7'][cmd.split()] > raw_path)()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='Study the behaviour of different tensor recovery methods.' )
    parser.add_argument( '--seed', type=int, default=23, help="A seed to generate seeds!" )
    parser.add_argument( '--execdir', type=str, default='state/execs/%s/'%(EXPT_NAME,), help="Location of the exec directories" )
    parser.add_argument( '--exptdir', type=str, default='state/expts/%s/'%(EXPT_NAME,), help="Location of the exec directories" )
    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser('run', help='Run the experiment' )
    run_parser.add_argument( '--pretend', action='store_true', help="Report how many settings you'd have" )
    run_parser.add_argument( '--parallel', action='store_true', help="Spawn parallel jobs?" )
    run_parser.add_argument( '--njobs', type=int, default=10, help="How many parallel jobs?" )
    run_parser.add_argument( '--instantiations', type=int, default=10, help="Number of different initial seeds to run with" )
    run_parser.add_argument( '--initializations', type=int, default=5, help="Number of different initial seeds to run with" )
    run_parser.add_argument( '--model', type=str, default="mixture", choices=["mixture","hmm","grid"], help="Model to use" )
    run_parser.add_argument( '--method', type=str, default="PowerMethod", choices=["PowerMethod","GradientPowerMethod"], help="Model to use" )
    #run_parser.add_argument( 'extra-args', type=str, nargs='+', help="Additional arguments for the actual program" )
    run_parser.set_defaults(func=do_run)

    plot_parser = subparsers.add_parser('process', help='Plot results from the experiment' )
    plot_parser.add_argument( '--model', type=str, default="mixture", choices=["mixture","hmm"], help="Model to use" )
    plot_parser.add_argument( '--best', action="store_true", help="When plotting, choose the best over the different runs." )
    plot_parser.set_defaults(func=do_process)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)

