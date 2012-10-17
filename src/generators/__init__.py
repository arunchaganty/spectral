"""
Data generators for various models
"""

from generators.MixtureModel import MixtureModel
from generators.GaussianMixtures import GaussianMixtureModel
from generators.TopicModel import TopicModel, LDATopicModel

import scipy as sc
from scipy import matrix, array
from scipy.linalg import norm 
import spectral.linalg as sl
import spectral.random as sr

def recovery_error( x, y ):
    """Return the difference between canonical forms of x and y"""
    x = array( x )
    y = array( y )
    x = sl.canonicalise( x )
    y = sl.canonicalise( y )

    return norm( x - y )

