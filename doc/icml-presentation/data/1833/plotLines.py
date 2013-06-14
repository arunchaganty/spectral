import numpy as np
import operator
import matplotlib
#matplotlib.use("Agg")

import matplotlib.pyplot as plt

def parseBetasFile(fname):
  betas = []

  beta = []
  for line in open(fname).xreadlines():
    if line.startswith("betas"):
      if( len(beta) > 0 ):
        betas.append(np.array(beta))
        beta = []
    else:
      try:
        beta.append( map(float, line.split()) )
      except Exception as e:
        print "[Warning]: Couldn't parse line", line
        print e
        continue
  if( len(beta) > 0 ):
    betas.append(np.array(beta))
  return betas

def fs153(x):
    return np.array([1.0, x, x**4, x**7])

def getLine( beta, fn, W ):
    """Gets a surface with X, Y, Z = f(X,Y) to be plotted"""
    X = np.linspace( -1, 1, W)
    # Grid of X's
    Y = map(lambda (x): beta.dot(fn(x)), X)

    return X, Y

def plotLines(ax, betas, modifier = "-", colors = None, W = 100):
    if colors is None:
        colors = ['b','r','g']
    colors = [ color + modifier for color in colors ]
    lines = []
    for beta, color in zip(betas.T, colors):
        X, Y = getLine( beta, fs153, W )
        lines += ax.plot(X, Y, color, linewidth=5)
    return lines

def makePlot( data, trueBetas, specBetas, spemBetas, em0Betas, emBetas, outFile ):

    plt.rc("font",size=20)

    for setting in [ [], ["EM"],["Spectral"],["Spectral+EM"],["Spectral","Spectral+EM"]]:
        fig = plt.figure()
        ax = plt.axes(xlim=(-1, 1), ylim=(-3, 5))
        ax.set_xlabel("t")
        ax.set_ylabel("y")

        # Plot the data
        if( len(setting) == 0 ):
            ax.scatter( data[0], data[1], alpha = 0.4 )
        else:
            ax.scatter( data[0], data[1], alpha = 0.4 )

        if "True" in setting:
            # Now draw each line
            plotLines( ax, trueBetas, "-" )

        # Legend 
        plottables = []
        # Spectral is dashed
        if( "Spectral" in setting ):
            lines = plotLines( ax, specBetas, "-", colors=['g']*3 )
            plottables.append( lines[0] )
        if( "Spectral+EM" in setting ):
            lines = plotLines( ax, spemBetas, "-", colors=['b']*3 )
            plottables.append( lines[0] )

        # EM is dash-dot
        if( "EM0" in setting ):
            plotLines( ax, em0Betas, ":" )
        if( "EM" in setting ):
            for emBeta in (emBetas[0],):
                lines = plotLines( ax, emBeta, "-", colors=['r']*3 )
            plottables.append( lines[0] )

        #for plot in plottables:
        #    plot.set_color( "black" )

        if( len(plottables) > 0):
            legend = plt.legend( plottables[::-1], setting[::-1] )
        #legend = plt.legend( plottables, ["Spectral", "Spectral + EM", "EM"] )
        #legend = plt.legend( plottables, ["EM"] )
        #legend = plt.legend( plottables, ["Spectral"] )
        #legend = plt.legend( plottables, ["Spectral", "Spectral + EM"] )


        prefix = "-".join(setting + [outFile])

        plt.savefig( prefix, transparent=True )
        plt.show()


def getBeta( specifier ):
  """ Get a particular beta"""
  if ":" in specifier:
      fname, idx = specifier.split(':')
      idx = int(idx)
  else:
      fname, idx = specifier, 0
  return parseBetasFile( fname )[idx]

def main( args ):
  try:
      data, trueBetas, specBetas, spemBetas, em0Betas, emBetas, outFile = args
  except:
      print "Usage: <data> <true-betas> <spec-betas>:idx <spec-em-betas>:idx <em0-betas>:idx <em-betas>:idx <out>"
      return

  # Data
  data = np.loadtxt( data )
  data = (data.T[1], data.T[-1])
  # Beta
  trueBetas = getBeta( trueBetas )

  specBetas = getBeta( specBetas )
  spemBetas = getBeta( spemBetas )

  em0Betas = getBeta( emBetas )
  #emBetas = getBeta( emBetas )
  emBetas = parseBetasFile( emBetas )

  makePlot( data, trueBetas, specBetas, spemBetas, em0Betas, emBetas, outFile )

if __name__ == "__main__":
  import sys
  main( sys.argv[1:] )


