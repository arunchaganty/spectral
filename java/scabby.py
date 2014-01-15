#!/usr/bin/env python2.7
#  

import sys, os
import itertools as it
from subprocess import Popen
import shlex

from collections import *

import numpy as np

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
        print( '%s ./%s'%( spawn_cmd, './'+batch_file ) )
        safe_run( '%s ./%s'%( spawn_cmd, './'+batch_file ) )

def dict_to_tab(data, sep='\t'):
    return sep.join( str(key) + '=' + str(val) for key, val in data.iteritems() )

def tab_to_dict(tab, sep='\t'):
    out = {}
    for item in tab.split(sep):
        key, value = item.split('=',1)
        out[key.strip()] = float(value.strip())
    return out

def align(data, key):
    arr = []
    for datum in data:
        arr.append( datum[key], datum )
    return arr

def to_matrix(data, keys):
    arr = []
    for datum in data:
        arr.append( [ datum[key] for key in keys ] )
    return np.array(arr)

def sort(data, key):
    arr = align(data, key)
    arr.sort()
    return [ val for key, val in arr ]

def list_to_dict( optlist, sep='=' ):
    if optlist is None: optlist = []
    kv = {}
    for item in optlist:
        key, val = item.split(sep,1)
        kv[key.strip()] = val.strip()
    return kv

def read_options(fname):
    options = {}
    for line in open(fname,'r'):
        if line == '': continue
        opts = line.split('\t', 1)
        if len(opts) > 1:
            key, val = line.split('\t', 1)
            options[key.strip()] = val.strip()
    return options

def fuzzy_get( dic, key ):
    for key_, val_ in dic.iteritems():
        if key_.endswith( key ):
            return val_
    else:
        raise KeyError

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

def aggregate( dicts, keys, mode = 'avg' ):
    values = {}
    updates = Counter()
    for val in dicts:
        key = tuple( (key, val[key]) for key in keys ) if keys else tuple(val.items())
        if key not in values:
            values[key], updates[key] = val, 1.
        else:
            if mode == 'avg':
                values[key], updates[key] = running_average( values[key], updates[key], val )
            elif mode == 'max':
                values[key], updates[key] = running_max( values[key], updates[key], val )
            elif mode == 'min':
                values[key], updates[key] = running_min( values[key], updates[key], val )
    return values.values()

def running_aggregate( agg, count, x, agg_fn  ):
    count += 1
    if isinstance( agg, dict ):
        for key in agg:
            agg[key] = agg_fn(agg[key], count, x[key])
    elif isinstance( agg, list ):
        for key in xrange(len(agg)):
            agg[key] = agg_fn(agg[key], count, x[key])

    return agg, count

def running_average( agg, count, x ):
    return running_aggregate( agg, count, x, 
            lambda agg, count, x: agg + float(x - agg)/count)
def running_max( agg, count, x ):
    return running_aggregate( agg, count, x, 
            lambda agg, count, x: max(x, agg))
def running_min( agg, count, x ):
    return running_aggregate( agg, count, x, 
            lambda agg, count, x: min(x, agg) )

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

def start_plot(**kwargs):
    import matplotlib.pyplot as plt 
    fig = plt.figure()
    ax = fig.gca()
    ax.set( **kwargs )
    return fig, ax

def plot(ax, data, x_key, y_key, **kwargs):
    xs, ys = zip(*[ (datum[x_key], datum[y_key]) for datum in data ] )
    ax.plot( xs, ys, **kwargs )
    return ax

def scatter(ax, data, x_key, y_key, **kwargs):
    opts = {'alpha':0.8}
    opts.update(kwargs)
    xs, ys = zip(*[ (datum[x_key], datum[y_key]) for datum in data ] )
    ax.scatter( xs, ys, **opts )
    return ax

def read_tab_file(fhandle):
    return (tab_to_dict(line) for line in fhandle)

def write_tab_file(dicts, out=sys.stdout):
    out.writelines(dict_to_tab(val) + "\n" for val in dicts)

def filter_tab(tab, **kwargs):
    def do_filter(point):
        return all( 
            # approximate match key and value
            any(key_.endswith( key ) and str(val_) == str(val) for key_, val_ in point.iteritems())
            for key, val in kwargs.iteritems())
    return filter( do_filter, tab )

import matplotlib.markers as markers
MARKERS = markers.MarkerStyle.filled_markers
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

