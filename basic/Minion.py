import numpy as np
from GridRegressor import *

class Minion:
    def __init__(self, dim, parent_grid, levels, training, labels, grid_refine_threshold, regparam):
        self.dim = dim
        self.parent_grid = parent_grid
        self.levels = levels
        self.training = training
        self.labels = labels
        self.grid_refine_threshold = grid_refine_threshold
        self.regparam = regparam

    def grid_regression(self, imax):
        grid_props = GridRegressor(self.dim, self.parent_grid, self.levels, self.regparam)
        print "Levels: ",self.levels
        grid_props.estimate_coefficients(self.training, self.labels, imax)

        if grid_props.grid_refine_ind > self.grid_refine_threshold:
            new_levels = self.levels
            for k in xrange(len(self.levels)):
                if self.levels[k] != 1:
                    new_levels = list(self.levels)
                    new_levels[k] += 1
                    new_levels = tuple(new_levels)
                    grid_props.grid_refinement_list.append(new_levels)

        return grid_props
