#!/usr/bin/env python2.7
# 
import os
import numpy as np
import matplotlib.pyplot as plt

def read_options(f):
    options = {}
    for line in open(f):
        k, v = line.split('\t')
        options[k] = v
    return options

def split_tab(line):
    items = line.split('\t')
    return map( lambda item: item.split('=')[1].strip(), items )
def tab_file_to_array(f):
    arr = []
    for line in open(f):
        arr.append( split_tab( line ) )
    return np.array( arr )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='plot the objective' )
    parser.add_argument( 'logdir', type=str, help="path to plot from" )
    args = parser.parse_args()

    options = read_options( os.path.join( args.logdir, 'options.map' ) )

    # Read from events file 
    events = tab_file_to_array( os.path.join(args.logdir, 'events') )
    def plot_events(events):
        events = np.array(events, dtype=np.float)
        plt.clf()
        plt.title("Outer iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Objective")
        plt.plot( events.T[0], events.T[1], 'ro-', label='e-objective' )
        plt.plot( events.T[0], -events.T[2], 'bo-', label='m-objective' )
        plt.legend()
        plt.grid()
        plt.show()
    plot_events(events)

    # Read from all iterations
    inner_events = []
    for i in xrange( int(options['Measurements.iters']) ):
        if os.path.exists( os.path.join( args.logdir, 'E-%d.events'%(i) ) ):
            events = tab_file_to_array( os.path.join( args.logdir, 'E-%d.events'%(i) ) )
            inner_events += list(np.array( events.T[1], dtype=np.double ))
            events = tab_file_to_array( os.path.join( args.logdir, 'M-%d.events'%(i) ) )
            inner_events += list(-np.array( events.T[1].T, dtype=np.double ))
    def plot_inner_events(events):
        plt.clf()
        plt.title("Inner iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Objective")
        plt.plot( events, 'ro-', label='objective' )
        plt.grid()
        plt.legend()
        plt.show()
    plot_inner_events(inner_events)

