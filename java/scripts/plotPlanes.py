import numpy as np
from mpl_toolkits.mplot3d import Axes3D
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

def fs(x,y):
    return np.array([1.0, x, y, x * y**3, x**2 * y**2, x**3 * y])

def getSurface( beta, fn, W ):
    """Gets a surface with X, Y, Z = f(X,Y) to be plotted"""
    u = np.linspace( -1, 1, W)
    # Grid of X's
    X, Y = np.meshgrid(u,u)
    Z = map(lambda (x,y): beta.dot(fn(x,y)), zip(X.flatten(),Y.flatten()))
    Z = np.array(Z).reshape( W, W )

    return X, Y, Z

def plotSurface(ax, X, Y, Z, color ):
    surf = ax.plot_surface(X, Y, Z, color = color, alpha = 0.7 )
def plotWireframe(ax, X, Y, Z, color ):
    surf = ax.plot_wireframe(X, Y, Z, color = color, alpha = 0.7 )

def plotSurfaces(ax, betas, colors = None, W = 100):
    if colors is None:
        colors = ['b','r','g']
    for beta, color in zip(betas.T, colors):
        X, Y, Z = getSurface( beta, W )
        plotSurface(ax, X, Y, Z, color)

def plotWireframes(ax, betas, colors = None, W = 100):
    if colors is None:
        colors = ['b','r','g']
    for beta, color in zip(betas.T, colors):
        X, Y, Z = getSurface( beta, W )
        plotWireframe(ax, X, Y, Z, color)

def getAxes():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    return ax

def plotDifferences( ax, betas, betas_, colors = None, W = 100):
    if colors is None:
        colors = ['b','r','g']
    for beta, beta_ in zip(betas.T,betas_.T):
        X, Y, Z = pp.getSurface(beta - beta_, 100)
        pp.plotSurface( ax, X, Y, Z, 'b')
        plt.show()
        plt.clf()
        ax = pp.getAxes()

def plotPoints( ax, X, Y, Z ):
    ax.scatter3D( X, Y, Z )

def plotAnimation( ax, X, Y, Z ):
    ax.scatter3D( X, Y, Z )

if __name__ == "__main__":
  import sys
  betas = parseBetasFile( sys.argv[1] )

  for beta in betas:
    print beta

