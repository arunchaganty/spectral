import numpy as np
import operator
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation 

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
    return np.array([1.0, x, x**4])

def getLine( beta, fn, W ):
    """Gets a surface with X, Y, Z = f(X,Y) to be plotted"""
    X = np.linspace( -1, 1, W)
    # Grid of X's
    Y = map(lambda (x): beta.dot(fn(x)), X)

    return X, Y

def plotLine(ax, X, Y, color):
    ax.plot(X, Y, color)

def plotLines(ax, betas, colors = None, W = 100):
    if colors is None:
        colors = ['b','r','g']
    for beta, color in zip(betas.T, colors):
        X, Y = getLine( beta, fs153, W )
        plotLine(ax, X, Y, color)

def plotAnimation( data, trueBetas, stepsBetas, out = "animation.mp4" ):
    """Plot an animation with betas being fixed, and betas_ coming from
    EM"""
    D, K = trueBetas.shape

    fig = plt.figure()

    ax = plt.axes(xlim=(-1, 1), ylim=(-3, 3))

    # Draw data
    ax.scatter( data[0], data[1], alpha=0.3 )
    # Draw the true betas
    plotLines( ax, trueBetas )

    # Draw dashed lines for the steps
    lines = reduce( operator.add, 
            [ax.plot([], [], color, lw=2) for (_, color) in zip(range(K), ["b--","r--","g--"])])

    # initialization function: plot the background of each frame
    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    # animation function.  This is called sequentially
    def animate(i):
        for beta, line in zip( stepsBetas[i].T, lines):
            X, Y = getLine( beta, fs153, 100 )
            line.set_data(X, Y)
        return lines

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(stepsBetas), interval=20, blit=False)
    anim.save(outFile, fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()

if __name__ == "__main__":
  import sys
  try:
      data, trueBetas, stepsBetas, outFile = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
  except:
      print "Usage: <data> <true-betas> <steps> <outfile>"
      sys.exit(1)

  # true betas
  data = np.loadtxt( data )
  data = (data.T[1], data.T[-1])
  # true betas
  trueBetas = parseBetasFile( trueBetas )[0]
  # steps betas
  stepsBetas = parseBetasFile( stepsBetas )

  plotAnimation( data, trueBetas, stepsBetas, outFile )


