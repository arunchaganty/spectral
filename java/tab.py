#!/usr/bin/env python2.7
# 
import os
import scabby
from collections import Counter

def do_list(args):
    filters = scabby.list_to_dict( args.filters ) if args.filters is not None else {}
    for execdir in scabby.get_execs( args.execdir, **filters ) :
        print execdir

def do_extract(args):
    filters = scabby.list_to_dict( args.filters ) if args.filters is not None else {}
    keys = args.keys
    for execdir in scabby.get_execs( args.execdir, **filters ) :
        try:
            values = {}
            opts = scabby.read_options( os.path.join( execdir, 'options.map' ) )
            out = scabby.read_options( os.path.join( execdir, 'output.map' ) )
            opts.update( out )
            values = { key : scabby.fuzzy_get(opts,key) for key in keys }
            print scabby.dict_to_tab(values)
        except KeyError:
            continue
        except IOError:
            continue

def do_agg(args):
    # Read each line, and aggegate
    scabby.write_tab_file( scabby.aggregate( scabby.read_tab_file(args.tab), args.keys, args.mode ) )

def do_sort(args):
    # Read each line, and sort on these indices
    data = scabby.read_tab_file(args.tab)
    scabby.write_tab_file( sorted( data, key = lambda datum: tuple(datum[key] for key in args.keys) ) )

def do_plot(args):
    import matplotlib.pyplot as plt 
    assert len( args.keys ) == 2 
    x_key, y_key = args.keys

    data = [ scabby.read_tab_file(tab) for tab in args.tabs ]
    labels = args.labels if args.labels is not None else range(len(data))

    plot_options = {'xlabel' : x_key, 'ylabel': y_key}
    plot_options.update( scabby.list_to_dict( args.plot_options ) )

    fig, ax = scabby.start_plot(**plot_options)
    for (datum, label, marker) in zip(data, labels, scabby.MARKERS):
        scabby.plot( ax, datum, x_key, y_key, label=label, marker=marker)
    ax.legend()
    if args.output is not None:
        plt.savefig(args.output)
    else:
        plt.show()

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser( description='Tab file swissknife' )

    subparsers = parser.add_subparsers()

    list_parser = subparsers.add_parser('list', help='Extract a tab file from a dir of execs' )
    list_parser.add_argument( '--execdir', type=str, help="Path to execution directory" )
    list_parser.add_argument( '--filters', type=str, nargs='*', help="Additional key=value filters" )
    list_parser.set_defaults(func=do_list)

    extract_parser = subparsers.add_parser('extract', help='Extract a tab file from a dir of execs' )
    extract_parser.add_argument( '--execdir', type=str, help="Path to execution directory" )
    extract_parser.add_argument( '--filters', type=str, nargs='*', help="Additional key=value filters" )
    extract_parser.add_argument( '--keys', type=str, nargs='+', help="Additional key=value filters" )
    extract_parser.set_defaults(func=do_extract)

    agg_parser = subparsers.add_parser('agg', help='Aggregate over a tab file' )
    agg_parser.add_argument( '--mode', default='avg', choices=['avg','max', 'min'], help="Tab file" )
    agg_parser.add_argument( '--tab', type=file, nargs='?', default=sys.stdin, help="Tab file" )
    agg_parser.add_argument( 'keys', nargs='*', help="Keys to aggregate on" )
    agg_parser.set_defaults(func=do_agg)

    sort_parser = subparsers.add_parser('sort', help='Sort by keys' )
    sort_parser.add_argument( '--tab', type=file, nargs='?', default=sys.stdin, help="Tab file" )
    sort_parser.add_argument( 'keys', nargs='+', help="Keys to sort on" )
    sort_parser.set_defaults(func=do_sort)

    plot_parser = subparsers.add_parser('plot', help='Plot by keys' )
    plot_parser.add_argument( '--plot-options', nargs='*', help="Options for plot; e.g. xlabel, ylabel" )
    plot_parser.add_argument( '--labels', nargs='+', help="Labels for each data file" )
    plot_parser.add_argument( '--keys', nargs='+', help="Keys to plot on (X, Y)" )
    plot_parser.add_argument( '--output', help="Where to save the plot" )
    plot_parser.add_argument( 'tabs', type=file, nargs='+', default=sys.stdin, help="Tab file" )
    plot_parser.set_defaults(func=do_plot)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)

