from __future__ import division
import sys
sys.path.append('/home/svn/sgpp/trunk/pysgpp')
sys.path.append('/home/karthikeya/MachineLearning/elastic_ml/sgct')
from tools import *
from pysgpp import *
from math import *
import random
from array import array
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

# Combination modules
from combinationScheme import *
import ActiveSetFactory

from scipy.optimize import minimize
from Minion import *
from CombiGridDict import *
from GridPriorityQueue import *
from GridConfig import *

class Master:
    def __init__(self, data, max_level, regparam, grid_refine_threshold, **kwargs):
        self.data = data
        self.max_level = max_level
        self.regparam = regparam
        self.grid_refine_threshold = grid_refine_threshold

        self.sgd_algo = kwargs.get('sgd_algo', 'vSGDfd')
        self.learning_rate = kwargs.get('learning_rate', 0.01)
        self.imax = kwargs.get('imax',100)

    def evaluate_cost_function(self, grid_results, grid_coeffs, labels):
        error = [np.dot(grid_result, grid_coeffs) for grid_result in grid_results] - labels
        error = error*error
        
        return  np.sum(error) / len(error)
    
    def estimate_grid_coefficients(self, grid_results, grid_coeffs_init, labels):
        func = lambda c: np.linalg.norm(np.sum(np.dot(g,c) for g in grid_results) - labels)
    
        cons = ({'type': 'eq', 'fun' : lambda c: np.sum(c) - 1})
    
        grid_coeffs = minimize(func, grid_coeffs_init, constraints=cons, method='SLSQP', options={'disp': False})
        print grid_coeffs
    
        return grid_coeffs.x

    def update_grid_config_queue(self, grid_pool, grid_config_queue, child_config_list, parent_levels):
        for item in child_config_list:
            print "grid_pool: ",grid_pool
            if item not in grid_pool:
                child_parent = []
                #The following order of append is important
                child_parent.append(item)
                child_parent.append(parent_levels)
                if item not in grid_config_queue:
                    priority = [sum(item), max(item)]
                    queue_obj = GridConfig(priority, child_parent)
                    grid_config_queue.put(queue_obj)

        return grid_config_queue
    
    def do_regression(self):    
        training = np.array(self.data["data"])
    
        labels = np.array(self.data["classes"])
        
        dim = training.shape[1]
        numData = training.shape[0]
    
        #if options.verbose:
        print "Dimension is:", dim
        print "Size of datasets is:", numData
        print "Level is: ", self.max_level
    
        lmin = tuple([2] * dim)
        lmax = tuple([self.max_level] * dim)
        factory = ActiveSetFactory.ClassicDiagonalActiveSet(lmax, lmin, 0)
        activeSet = factory.getActiveSet()
        scheme = combinationSchemeArbitrary(activeSet)
        init_scheme = scheme.dictOfScheme
        print "Initial list of Sub grids: ",init_scheme
    
        combi_grid_dict = {}
        grid_config_queue = GridPriorityQueue()
        grid_coeffs_init = []
        for key, value in scheme.dictOfScheme.iteritems():
            grid_coeffs_init.append(value)
     
            minion = Minion(dim, None, key, training, labels, self.grid_refine_threshold, self.regparam)
            grid_props, child_config_list = minion.grid_regression(self.imax)
    
            combi_grid_dict[key] = grid_props
            if child_config_list:
                grid_config_queue = self.update_grid_config_queue(combi_grid_dict.keys(), grid_config_queue, child_config_list, key)

        grid_results = []                                     # u vector
        for key, values in combi_grid_dict.iteritems():
            grid_results.append(values.grid_result)
            print key,":  ",values.mse
    
        while not grid_config_queue.empty():
            priority_item = grid_config_queue.get()
            child_parent = priority_item.value
            child_levels = child_parent[0]
            parent_levels = child_parent[1]
            parent_grid_props = combi_grid_dict[parent_levels]
            minion = Minion(dim, parent_grid_props, child_levels, training, labels, self.grid_refine_threshold, self.regparam)
            grid_propsi, child_config_list = minion.grid_regression(self.imax)
            combi_grid_dict[child_levels] = grid_props
            if child_config_list:
                grid_config_queue = self.update_grid_config_queue(combi_grid_dict.keys(), grid_config_queue, child_config_list, child_levels) 

        grid_results = []                                     # u vector
        for key, values in combi_grid_dict.iteritems():
            grid_results.append(values.grid_result)
            print key,":  ",values.mse
    
        grid_results = np.array(grid_results).T
    
        #grid_coeffs = self.estimate_grid_coefficients(grid_results, grid_coeffs_init, labels)   # w vector
      
        #error = evaluate_cost_function(grid_results, grid_coeffs, labels)
    
        #print "Cost Function MSE: ", error
    
        print combi_grid_dict
    
        return combi_grid_dict #grid_coeffs
