from __future__ import division
import sys
sys.path.append('/home/svn/sgpp/trunk/pysgpp')
sys.path.append('/home/karthikeya/MachineLearning/elastic_ml/sgct')
from tools import *
from pysgpp import *
from math import *
import random
from optparse import OptionParser
from array import array

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
import scipy.io.arff as arff

# Combination modules
from combinationScheme import *
import ActiveSetFactory

from scipy.optimize import minimize

from GridFactory import *
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

def evaluate_cost_function(grid_results, grid_coeffs, labels):
    error = [np.dot(grid_result, grid_coeffs) for grid_result in grid_results] - labels
    error = error*error
    
    return  np.sum(error) / len(error)

def estimate_grid_coefficients(grid_results, grid_coeffs_init, labels):
    func = lambda c: np.linalg.norm(np.sum(np.dot(g,c) for g in grid_results) - labels)

    cons = ({'type': 'eq', 'fun' : lambda c: np.sum(c) - 1})

    grid_coeffs = minimize(func, grid_coeffs_init, constraints=cons, method='SLSQP', options={'disp': False})
    print grid_coeffs

    return grid_coeffs.x

#-------------------------------------------------------------------------------

def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z

def refine_grids(fullgrid_dict, refinement_threshold, training, labels, i):
    final_dict = fullgrid_dict.copy()
    for key, value in fullgrid_dict.iteritems():
        fg_subdict = {}
        if value.mse > refinement_threshold and np.sum(key) != value.dim and i < 10:
            levels = key
            print "Parent: ",key
            for k in xrange(len(levels)):
                if levels[k] != 1:
                    levels = list(levels)
                    levels[k] += 1
                    levels = tuple(levels)
                    print "Child: ",levels
                    #grid_obj = GridFactory(value.dim, value.alpha, levels, options.regparam)
                    #grid_obj.estimate_coefficients(training, labels)         
                    #fg_subdict[levels] = grid_obj
                    fg_subdict = refine_grids(fg_subdict, refinement_threshold, training, labels, i + 1)
            del final_dict[key]
            final_dict = merge_two_dicts(final_dict, fg_subdict)

    return final_dict


def do_regression():
    
    regparam = options.regparam

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

    lmin = tuple([1] * dim)
    lmax = tuple([5] * dim)
    factory = ActiveSetFactory.ClassicDiagonalActiveSet(lmax, lmin, 0)
    activeSet = factory.getActiveSet()
    scheme = combinationSchemeArbitrary(activeSet)
    init_scheme = scheme.dictOfScheme
    print "Initial list of Sub grids: ",init_scheme

    fullgrid_dict = {}
    grid_coeffs_init = []
    for key, value in scheme.dictOfScheme.iteritems():
        grid_coeffs_init.append(value)

        alpha = None
        grid_obj = GridFactory(dim, alpha, key, regparam)
        grid_obj.estimate_coefficients(training, labels) 

        fullgrid_dict[key] = grid_obj

    grid_results = []                                     # u vector
    for key, values in fullgrid_dict.iteritems():       
        grid_results.append(values.grid_result)
        print key,":  ",values.mse

    fullgrid_dict = refine_grids(fullgrid_dict, 0.001, training, labels, 0)

    grid_results = np.array(grid_results).T

    grid_coeffs = estimate_grid_coefficients(grid_results, grid_coeffs_init, labels)   # w vector
  
    error = evaluate_cost_function(grid_results, grid_coeffs, labels)

    print "Cost Function MSE: ", error

    return fullgrid_dict, grid_coeffs

#-------------------------------------------------------------------------------

if __name__=='__main__':
        # Initialize OptionParser, set Options
        parser = OptionParser()
        parser.add_option("-l", "--level", action="store", type="int", dest="level", help="Gridlevel")
        parser.add_option("-m", "--mode", action="store", type="string", default="apply", dest="mode", help="Specifies the action to do. Get help for the mode please type --mode help.")
        parser.add_option("-L", "--lambda", action="store", type="float",default=1E-5, metavar="LAMBDA", dest="regparam", help="Lambda")
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
