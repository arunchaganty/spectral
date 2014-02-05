#!/usr/bin/env python2.7
"""
Plot Jacobian singular value vs. error.
"""

import os
import random
import numpy as np
import scabby

EXPT_NAME = "PairwiseHMMRecovery"

KD_VALUES = [(2,3),]# (3,10), (5,10)]
N_VALUES = [1e3, 2e3, 5e3, 7e3, 
            1e4, 2e4, 5e4, 7e4, 
            1e5, 2e5, 5e5, 7e5, 
            1e6, 
            5e7 # equivalent to infinity
            ]
MODES = ["EM", "Piecewise", "Spectral"]

def get_settings(args):
    l = 5
    for mode in MODES:
        for k,d in KD_VALUES:
            assert k <= d
            for n in N_VALUES:
                for model_seed in xrange( args.instantiations ):
                    for data_seed in xrange( args.realizations ):
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
 -mode {mode} -N {n} \
 -K {k} -D {d} -L {l}\
 -smoothMeasurements 1e-4\
 -paramsRandom {model_seed}\
 -genRandom {data_seed}\
 -initRandom {init_seed}'

    settings = get_settings(args)

    if args.pretend:
        count = len(list(settings))
        print "%d settings split over %d runs (%d each)" %(count, args.njobs, count/args.njobs+1)
    elif args.parallel:
        if args.nlpsub:
            scabby.parallel_spawn( args.exptdir, "qstart-short", cmd, args.njobs, settings )
        else:
            scabby.parallel_spawn( args.exptdir, "echo", cmd, args.njobs, settings )
    else:
        for setting in settings:
            scabby.safe_run(cmd.format(**setting))

def do_process(args):
    import plumbum as pb

    def tab(args):
        return pb.local['python2.7'][ args.split() ] 
    cat = pb.local['cat']

    for k, d in KD_VALUES:
        for mode in MODES:
            print k, d
            cmd = 'tab.py extract \
     --execdir {args.execdir} \
     --filters K={k} D={d} mode={mode}\
     --keys N paramsRandom initRandom genRandom L O-sigmak O-K piecewise-sigmak dO dT dPi dtheta0'.format(**locals())
            raw_path = os.path.join( args.exptdir, 'Pairwise-{k}-{d}-{mode}.raw.tab'.format(**locals()) )
            cmd = tab(cmd) | tab("tab.py sort paramsRandom initRandom genRandom N") > raw_path
            print cmd
            cmd()

            # Aggregate
            agg_path = os.path.join( args.exptdir, 'Pairwise-{k}-{d}-{mode}.agg.tab'.format(**locals()) )
            cmd = cat[raw_path] | tab('tab.py agg --mode min N paramsRandom') | tab("tab.py sort paramsRandom N") > agg_path
            print cmd
            cmd()

            macro_path = os.path.join( args.exptdir, 'Pairwise-{k}-{d}-{mode}.macro.tab'.format(**locals()) )
            cmd = cat[agg_path] | tab('tab.py agg N') | tab("tab.py sort N") > macro_path
            print cmd
            cmd()



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
    run_parser.add_argument( '--nlpsub', action='store_true', help="Just make scirpts" )
    run_parser.add_argument( '--njobs', type=int, default=10, help="How many parallel jobs?" )
    run_parser.add_argument( '--instantiations', type=int, default=10, help="Number of different initial seeds to run with" )
    run_parser.add_argument( '--realizations', type=int, default=3, help="Number of different initial seeds to run with" )
    run_parser.add_argument( '--initializations', type=int, default=3, help="Number of different initial seeds to run with" )
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

