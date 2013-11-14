#!/usr/bin/env python2.7
#  

import os
import itertools as it
from subprocess import Popen
import shlex

def safe_run( cmd, block = True ):
    proc = Popen( shlex.split( cmd ) )
    if block:
        proc.wait()
    return proc

def parallel_spawn( exptdir, spawn_cmd, cmd, n_jobs, settings ):
    """Spawn command interpreted with **kwargs"""
    settings = list(settings)
    batch_size = len(settings) / n_jobs
    for i in xrange(n_jobs):
        start, end = batch_size * i, min( batch_size * (i+1), len(settings))
        batch_file = os.path.join(exptdir, 'batch-%d.sh'%(i))
        with open(batch_file, 'w') as script:
            for setting in settings[start:end]:
                script.write( cmd.format(**setting) + "\n" )
        safe_run( 'chmod +x %s'%(batch_file) )
        safe_run( '%s ./%s'%( spawn_cmd, './'+batch_file ) )

def read_options(fname):
    options = {}
    for line in open(fname,'r'):
        if line == '': continue
        key, val = line.split('\t', 1)
        options[key.strip()] = val.strip()
    return options

def get_execs( root_dir, **kwargs ):
    """Get all executions which match the options in **kwargs"""

    for exec_dir in os.listdir( root_dir ):
        exec_dir = os.path.join( root_dir, exec_dir )
        # Make sure this is a directory
        if not os.path.isdir( exec_dir ): continue
        # Now check that it has an options
        if not "options.map" in os.listdir( exec_dir ): continue
        # Finally, check that we have all the options you asked for
        options_map = os.path.join(exec_dir, "options.map")
        options = read_options( options_map )
        # Check that we have all your keys.
        if all(
                # approximate match key and value
                any(key_.endswith( key ) and val_ == str(val) for key_, val_ in options.iteritems())
            for key, val in kwargs.iteritems()):
            yield exec_dir

def running_average( avg, count, x ):
    count += 1
    for i in xrange(len(avg)):
        avg[i] += float(x[i] - avg[i])/count
    return avg, count

def plot_many(lines, xlabel='x', ylabels=['y']):
    """
    Assumes lines is a list of np arrays
    """
    import matplotlib.pyplot as plt 
    import matplotlib.markers as markers
    figs = []
    for i, ylabel in enumerate(ylabels):
        fig = plt.figure(i)
        fig.clear()
        ax = fig.gca()
        ax.set_ylabel(ylabel)
        for (legend, line), marker in zip(lines, markers.MarkerStyle.filled_markers):
            if len(line) == 0: continue
            ax.plot( line.T[0], line.T[i+1], label=str(legend), marker=marker )
        ax.legend()
        ax.grid()

        figs.append(fig)
    return figs

