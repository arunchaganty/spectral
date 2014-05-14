#!/usr/bin/env python2.7
"""

"""

import scabby
import numpy as np
import matplotlib.pyplot as plt

def do_command(args):
    plt.rc('font', family = "Serif", size=20)
    plt.rc('axes', labelsize = 24)
    plt.rc('xtick', labelsize = 24)
    plt.rc('ytick', labelsize = 24)
    plt.rc('text', usetex = True)

    data = scabby.tab_to_table(scabby.read_tab_file(args.fname), 'noise', 'dZ_pi', 'std_dZ_pi', 'dZ_cl', 'std_dZ_cl')
    print data
    plt.errorbar( data.T[0], data.T[1], yerr=data.T[2], linestyle='-', marker='o', label='Pseudoinverse')
    plt.errorbar( data.T[0], data.T[3], yerr=data.T[4], linestyle='-', marker='s', label='Composite-Likelihood')

    plt.ylabel(r'$\|\theta - \hat \theta\|_2$')
    plt.xlabel('$\epsilon$')

    plt.xscale('log')
    plt.yscale('log')
    plt.gca().invert_xaxis()

    plt.legend(loc='lower left')

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output + ".pdf")
        plt.savefig(args.output + ".png")
    plt.show()
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='' )
    parser.add_argument( 'fname', type=file, help="table to plot" )
    parser.add_argument( 'output', type=str, default=None, help="Where to output" )
    parser.set_defaults(func=do_command)

    #subparsers = parser.add_subparsers()
    #command_parser = subparsers.add_parser('command', help='' )
    #command_parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)
