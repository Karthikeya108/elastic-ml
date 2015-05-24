import sys
sys.path.append('/home/karthikeya/svn/repo/lib/pysgpp')
sys.path.append('/home/karthikeya/MachineLearning/elastic_ml/pyoptim')
from tools import *
from pysgpp import *
from math import *
import random
from optparse import OptionParser
from array import array

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from wrappers import SparseGridWrapper
import seaborn as sns
import scipy.io.arff as arff
from algorithms import vSGD, SGD, vSGDfd, AnnealingSGD, AveragingSGD

#-------------------------------------------------------------------------------
## Formats a list of type mainlist as a string
# <pre>
#     main_list  = {plain_list | string}*
#     plain_list = string*
# </pre>
# @param l Mainlist
def format_optionlist(l):
    def join_inner_list(entry):
        if type(entry) is list:
            return "("+' OR '.join(entry)+")"
        else:
            return entry
    return ' '*4+' AND '.join(map(lambda entry: join_inner_list(entry), l))


#-------------------------------------------------------------------------------
## Checks whether a valid mode is specified,
# whether all required options for the mode are given and
# executes the corresponding action (function)
#
# @todo remove hack for level new when porting to classifier.new.py
#
# @param mode current mode
def exec_mode(mode):

    if mode=="help":
        print "The following modes are available:"
        for m in modes.keys():
            print "%10s: %s" % (m, modes[m]['help'])
        sys.exit(0)

    # check valid mode
    if not modes.has_key(mode):
        print("Wrong mode! Please refer to --mode help for further information.")
        sys.exit(1)

    # check if all required options are set
    a = True
    for attrib in modes[mode]['required_options']:
        # OR
        if type(attrib) is list:
            b = False
            for attrib2 in attrib:
                # hack for level 0
                if attrib2 == "level":
                    option = getattr(options, attrib2, None)
                    if option >= 0:
                        b = True
                else:
                    if getattr(options, attrib2, None):
                        b = True
            a = a and b
        else:
            if not getattr(options, attrib, None):
                a = False
    if not a:
        print ("Error!")
        print ("For --mode %s you have to specify the following options:\n" % (mode)
               + format_optionlist(modes[mode]['required_options']))
        print ("More on the usage of %s with --help" % (sys.argv[0]))
        sys.exit(1)

    # execute action
    modes[mode]['action']()


#-------------------------------------------------------------------------------
## Opens and read the data of an ARFF (or plain whitespace-separated data) file.
# Opens a file given by a filename.
# @param filename filename of the file
# @return the data stored in the file as a set of arrays
def openFile(filename):
    if "arff" in filename:
        return readData(filename)
    else:
        return readDataTrivial(filename)

def evaluate_error(labels, alpha, design_matrix):
    error = DataVector(len(labels))
    design_matrix.mult(DataVector(alpha), error)
    error.sub(labels) 
    error.sqr() 
    mse = error.sum() / float(len(error))
    print "MSE: %g on %d data pts" % (mse, len(labels))
    print "RMSE / L2-norm of error on data: %g" % (sqrt(mse))

    return mse

def collect_mse(s, y, collection, mult_eval):
    print s.provider._counter
    if s.provider._counter % 10 == 0:
        params_DV = DataVector(s.bestParameters)
        results_DV = DataVector(len(y))
        mult_eval.mult(params_DV, results_DV)
        residual = np.linalg.norm(y - results_DV.array())
        collection.append([s.bestParameters, s.provider._counter, np.mean(s.learning_rate), residual**2/len(y)])

def run(grid, alpha, training, labels):
    grid_size = grid.getStorage().size()

    data = np.vstack((training.T,labels)).T

    '''
    if options.regstr == 'identity':
        opL = createOperationIdentity(grid)
    elif options.regstr == 'laplace':
        opL = createOperationLaplace(grid)
    '''
    print "Forming the Desing Matrix" 
    design_matrix = createOperationMultipleEval(grid, DataMatrix(training))

    l = 1E-8

    print "Creating Sparse Grid Wrapper"
    f = SparseGridWrapper(grid, data, l)
    x0 = np.zeros(grid_size)

    collect_fd = []
    collect_cb = lambda s: collect_mse(s, labels, collect_fd, design_matrix)
    f.reset()
    algo = vSGDfd(f, x0, callback=collect_cb, loss_target=-np.inf, init_samples=1, batch_size=10)
    print "Running SGD"
    algo.run(100)

    collect_fd = np.array(collect_fd)

    info_record = collect_fd[collect_fd.shape[0]-1,:]

    print "Itearation %d, learning rate %f, train MSE %e" %(info_record[1], info_record[2], info_record[3])

    alpha = info_record[0]
    
    error = evaluate_error(DataVector(labels), DataVector(alpha), design_matrix)

    return grid, DataVector(alpha), error

#-------------------------------------------------------------------------------

def do_regression():
    # read data
    data = openFile(options.data[0])
 
    training = np.array(data["data"])

    labels = np.array(data["classes"])
    
    dim = training.shape[1]
    numData = training.shape[0]

    level = options.level

    #if options.verbose:
    print "Dimension is:", dim
    print "Size of datasets is:", numData
    print "Level is: ", options.level

    grid = Grid.createLinearGrid(dim)
    generator = grid.createGridGenerator()
    generator.regular(level)

    gsize = grid.getSize()
    newGsize = 0

    print "Gridsize is:", gsize

    alpha = DataVector(grid.getSize())
    alpha.setAll(1.0)

    grid, alpha, error = run(grid, alpha, training, labels)

    return alpha

#-------------------------------------------------------------------------------

if __name__=='__main__':
        # Initialize OptionParser, set Options
        parser = OptionParser()
        parser.add_option("-l", "--level", action="store", type="int", dest="level", help="Gridlevel")
        parser.add_option("-m", "--mode", action="store", type="string", default="apply", dest="mode", help="Specifies the action to do. Get help for the mode please type --mode help.")
        parser.add_option("-L", "--lambda", action="store", type="float",default=0.01, metavar="LAMBDA", dest="regparam", help="Lambda")
        parser.add_option("-R", "--regstr", action="store", type="string",default='identity', metavar="REGSTR", dest="regstr", help="RegStrategy")
        parser.add_option("-i", "--imax", action="store", type="int",default=500, metavar="MAX", dest="imax", help="Max number of iterations")
        parser.add_option("-d", "--data", action="append", type="string", dest="data", help="Filename for the Datafile.")
        parser.add_option("-t", "--test", action="store", type="string", dest="test", help="File containing the testdata")
        parser.add_option("-A", "--alpha", action="store", type="string", dest="alpha", help="Filename for a file containing an alpha-Vector")
        parser.add_option("-o", "--outfile", action="store", type="string", dest="outfile", help="Filename where the calculated alphas are stored")
        parser.add_option("--gridfile", action="store", type="string", dest="gridfile", help="Filename where the resulting grid is stored")
        parser.add_option("-v", "--verbose", action="store_true", default=False, dest="verbose", help="Provides extra output")
        parser.add_option("--grid", action="store", type="string", dest="grid", help="Filename for Grid-resume. For fold? and test. Full filename.")
        parser.add_option("--mse_limit", action="store", type="float", default="0.0", dest="mse_limit", help="If MSE of test data fall below this limit, refinement will stop.")
        parser.add_option("--grid_limit", action="store", type="int", default="0", dest="grid_limit", help="If the number of points on grid exceed grid_limit, refinement will stop.")
        parser.add_option("--Hk", action="store", type="float", default="1.0", dest="Hk", help="Parameter k for regularization with H^k norm. For certain CModes.")

        parser.add_option("--function_type", action="store", type="choice", default=None, dest="function_type", choices=['modWavelet'],
                      help="Choose type for non-standard basis functions")
    # parse options
        (options,args)=parser.parse_args()

    # specifiy the modes:
        # modes is an array containing all modes, the options needed by the mode and the action
        # that is to be executed
        modes = {
                 'regression'   : {'help': "learn a dataset",
                      'required_options': ['data', 'level'],
                      'action': do_regression}
                }

    
        exec_mode(options.mode.lower())
