from __future__ import division
import sys
sys.path.append('/home/svn/sgpp/trunk/pysgpp')
sys.path.append('/home/karthikeya/MachineLearning/elastic_ml/pyoptim')
from tools import *
from pysgpp import *
from math import *
import random
from optparse import OptionParser

import numpy as np

import matplotlib.pyplot as plt

from wrappers import SparseGridWrapper
from algorithms import vSGD, SGD, vSGDfd, AnnealingSGD, AveragingSGD

from FullGrid import FullGrid
from scipy.optimize import minimize

class GridFactory:
    def __init__(self, dim, alpha, levels, regparam):
        self.levels = levels
        self.regparam = regparam
        self.dim = dim

        self.grid = FullGrid(self.dim, self.levels)

        if alpha == None:
            self.alpha = alpha
        else:
            self.alpha = alpha  #TODO

        self.grid_refine_ind = 0
        self.grid_result = [None] * self.grid.get_size()
        self.mse = 0
    
    def compute_grid_refinement_indicator(self, grid_result, labels):
        error = grid_result - labels
        error = error*error

        return  np.sum(error) / (len(labels) - 1)

    def evaluate_error(self, data, labels):
        design_matrix = self.grid.evaluate(DataMatrix(data))
        error = DataVector(len(labels))
        error = DataVector(np.dot(design_matrix.array(), self.alpha.array()))
        error.sub(labels)
        error.sqr()
        mse = error.sum() / float(len(error))
        print "MSE: %g on %d data pts" % (mse, len(labels))
        print "RMSE / L2-norm of error on data: %g" % (sqrt(mse))

        return mse

    def collect_mse(self, s, y, collection, mult_eval):
        print s.provider._counter
        if s.provider._counter % 10 == 0:
            params_DV = s.bestParameters
            results_DV = np.dot(mult_eval.array(), params_DV)
            residual = np.linalg.norm(y - results_DV)
            collection.append([s.bestParameters, results_DV, s.provider._counter, np.mean(s.learning_rate), residual**2/len(y)])
 
    def estimate_coefficients(self, train_data, labels):
        grid_size = self.grid.get_size()

        data = np.vstack((train_data.T,labels)).T

        print "Forming the Desing Matrix"
        design_matrix = self.grid.evaluate(DataMatrix(train_data))

        print "Creating Sparse Grid Wrapper"
        f = SparseGridWrapper(self.grid, data, self.regparam)
        x0 = np.zeros(grid_size)

        collect_fd = []
        collect_cb = lambda s: self.collect_mse(s, labels, collect_fd, design_matrix)
        f.reset()
        algo = vSGDfd(f, x0, callback=collect_cb, loss_target=-np.inf, init_samples=1, batch_size=10)
        print "Running SGD"
        algo.run(10)

        collect_fd = np.array(collect_fd)

        info_record = collect_fd[collect_fd.shape[0]-1,:]

        print "Itearation %d, learning rate %f, train MSE %e" %(info_record[2], info_record[3], info_record[4])

        self.alpha = info_record[0]

        self.grid_result = info_record[1]

        self.grid_refine_ind = self.compute_grid_refinement_indicator(self.grid_result, labels)

        self.mse = info_record[4]

        return

    
