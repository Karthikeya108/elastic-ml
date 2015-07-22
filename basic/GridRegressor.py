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

class GridRegressor:
    def __init__(self, dim, parent_grid_props, levels, regparam):
        self.levels = levels
        self.regparam = regparam
        self.dim = dim

        self.grid = FullGrid(self.dim, self.levels)

        if parent_grid_props is None:
            self.alpha = DataVector(self.grid.get_size())
            self.alpha.setAll(1.0)
        else:
            self.coefficient_interpolation(parent_grid_props)

        self.grid_refine_ind = 0
        self.grid_result = [None] * self.grid.get_size()
        self.mse = 0
   
    def coefficient_interpolation(self, parent_grid_props):
        self.alpha = DataVector(self.grid.get_size())
        self.alpha.setAll(1.0)

        parent_levels = parent_grid_props.levels
        for i in xrange(len(self.levels)):
             if self.levels[i] > parent_levels[i]:
                  refine_dim = i + 1

        parent_alpha = parent_grid_props.alpha
        parent_grid = parent_grid_props.grid

        coordinates = self.grid.get_coordinates()
        parent_coordinates = parent_grid.get_coordinates()

        unit_dist = 1.0 / 2**self.levels[refine_dim - 1]
        print "unit_dist: ",unit_dist
        parent_alpha[-1] = 0
        for coords, index in coordinates.iteritems():
            if coords in parent_coordinates:
                parent_index = parent_coordinates[coords]
                self.alpha[index] = parent_alpha[parent_index]
            else:
                coords_left = list(coords)
                coords_right = list(coords)
                coords_left[refine_dim - 1] = coords_left[refine_dim - 1] - unit_dist
                coords_right[refine_dim - 1] = coords_right[refine_dim - 1] + unit_dist
                if tuple(coords_left) in parent_coordinates:
                     alpha_left_index = parent_coordinates[tuple(coords_left)]
                else:
                     alpha_left_index = -1
                if tuple(coords_right) in parent_coordinates:
                     alpha_right_index = parent_coordinates[tuple(coords_right)]
                else:
                     alpha_right_index = -1

                self.alpha[index] = np.mean([parent_alpha[alpha_left_index], parent_alpha[alpha_right_index]])

        print "Parent alpha: ",DataVector(parent_alpha)
        print "Alpha :",self.alpha
                    
    def compute_grid_refinement_indicator(self, grid_result, labels):
        error = grid_result - labels
        error = error*error

        return  np.sum(error) / (len(labels) - 1)

    def evaluate_error(self, data, labels):
        design_matrix = self.grid.evaluate(DataMatrix(data))
        error = DataVector(len(labels))
        error = DataVector(np.dot(design_matrix.array(), self.alpha))
        error.sub(DataVector(labels))
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
 
    def estimate_coefficients(self, train_data, labels, imax):
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
        algo = vSGDfd(f, x0, callback=collect_cb, loss_target=-np.inf, init_samples=10, batch_size=10)
        #algo = SGD(f, x0, callback=collect_cb, loss_target=-np.inf, learning_rate=0.001, batch_size=10)
        print "Running SGD"
        algo.run(imax)

        collect_fd = np.array(collect_fd)

        info_record = collect_fd[collect_fd.shape[0]-1,:]

        print "Itearation %d, learning rate %f, train MSE %e" %(info_record[2], info_record[3], info_record[4])

        self.alpha = info_record[0]

        self.grid_result = info_record[1]

        self.grid_refine_ind = self.compute_grid_refinement_indicator(self.grid_result, labels)

        self.mse = self.evaluate_error(train_data, labels)

        return

    
