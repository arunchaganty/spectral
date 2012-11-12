# Plot cool statistics about data 

import ipdb
import os
import re
import scipy as sc
import spectral.MultiView as mv
import spectral.SphericalGaussians as sg
import matplotlib.pyplot as plt

def plot_time_err( out, em_err, spec_err ):
    em_err.sort()
    em_err = sc.array( em_err ).real
    spec_err = sc.array( spec_err ).real

    plt.clf()
    plt.plot( em_err.T[0], em_err.T[1], c='r', label="EM" )
    plt.plot( spec_err.T[0], spec_err.T[1], c='b', marker="+", label="Spec" )

    plt.legend()
    plt.savefig( out )

def plot_total_err( out, em_err, spec_err ):
    em_err = sc.array( em_err ).real
    spec_err = sc.array( spec_err ).real

    plt.clf()
    plt.scatter( spec_err.T[0], spec_err.T[1], s = 100*spec_err.T[2], color='b', alpha=0.5, label="Spec" )
    plt.scatter( em_err.T[0], em_err.T[1], s = 100*em_err.T[2], color='r', alpha=0.5, label="EM" )
    plt.savefig( out )

def plot_bound_err( out, spec_err ):
    spec_err = sc.array( spec_err ).real

    n, k = spec_err.shape 
    for i in xrange( k-2):
        plt.clf()
        plt.scatter( spec_err.T[0], spec_err.T[1], s = 100*spec_err.T[2+i], color='b', alpha=0.5, label="Spec" )
        plt.legend()
        plt.savefig( out%i )

def load_data( *args ):
    """Load data from these dataset files"""

    collector_total_err_em = []
    collector_total_err_spec = []
    collector_bound_err_spec = []

    em_rex = re.compile( r"mvgmm-(?P<k>\d+)-(?P<d>\d+)-(?P<s>\d+\.\d+)(_\d+)?-em-1e6-1e6" )
    spec_rex = re.compile( r"mvgmm-(?P<k>\d+)-(?P<d>\d+)-(?P<s>\d+\.\d+)(_\d+)?-spec-1e6-1e6" )

    em_t_rex = re.compile( r"aerr_M3_t(?P<t>\d+)" )

    # TODO: Specialise by sigma^2, N
    for ds in args:
        log_path = os.path.join("results", os.path.dirname( ds ) )
        prefix = os.path.basename(ds)[:-4] # Get rid of the .npz
        collector_time_err_em = []
        collector_time_err_spec = []
        for logf in os.listdir( log_path ):
            logf_ = os.path.join(log_path, logf)
            # Filter the logs which have what we want
            if not logf.startswith( prefix ):
                continue
            if em_rex.match( logf ):
                l = sc.load( logf_ )
                if "time" in l.keys():
                    collector_total_err_em.append( ( l["k"], l["d"], l["aerr_M3_col"] ) )
                    for k in l.keys():
                        if em_t_rex.match( k ):
                            t = int( em_t_rex.match( k ).groupdict()['t'] )
                            collector_time_err_em.append( (l["time_%d" % t], l[k]/sc.sqrt(l["k"]) ) ) # Approximating the min col-norm
                    #collector_time_err_em.append( (l["time"], l["aerr_M_col"]) )
            elif spec_rex.match( logf ):
                print logf
                l = sc.load( logf_ )
                if "time" in l.keys():
                    collector_total_err_spec.append( (l["k"], l["d"], l["aerr_M3_col"] ))
                    collector_time_err_spec.append( (l["time"], l["aerr_M3_col"]) )
                    bounds = mv.compare_error_bounds( ds, logf_ )
                    collector_bound_err_spec.append( [l["k"], l["d"]] + bounds )
            else:
                pass

        if len( collector_time_err_spec ) > 0 and len( collector_time_err_em ) > 0:
            plot_time_err( "mvgmm/" + prefix + ".jpg", collector_time_err_em, collector_time_err_spec )
    plot_total_err( "total.jpg", collector_total_err_em, collector_total_err_spec )
    plot_bound_err( "bound%i.jpg", collector_bound_err_spec )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='F', nargs='+', help='files to parse')
    args = parser.parse_args()
    load_data( *args.files )

