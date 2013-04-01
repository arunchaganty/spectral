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
        lines += ax.plot(X, Y, color, linewidth=2)
    return lines

def makePlot( data, trueBetas, specBetas, spemBetas, em0Betas, emBetas, outFile ):
    fig = plt.figure()
    ax = plt.axes(xlim=(-1, 1), ylim=(-3, 5))

    # Plot the data
    ax.scatter( data[0], data[1], alpha = 0.1 )

    # Now draw each line
    #plotLines( ax, trueBetas, "-" )

    # Legend 
    plottables = []
    # Spectral is dashed
    lines = plotLines( ax, specBetas, "--" )
    plottables.append( lines[0] )
    lines = plotLines( ax, spemBetas, "-" )
    plottables.append( lines[0] )

    # EM is dash-dot
    #plotLines( ax, em0Betas, ":" )
    #for emBeta in emBetas:
    #    lines = plotLines( ax, emBeta, ":" )
    #plottables.append( lines[0] )

    for plot in plottables:
        plot.set_color( "black" )

    #plt.legend( plottables, ["Spectral", "Spectral + EM", "EM"] )
    plt.legend( plottables, ["Spectral", "Spectral + EM", "EM"] )

    #plt.savefig(outFile)
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

