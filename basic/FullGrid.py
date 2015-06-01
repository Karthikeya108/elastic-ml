import sys, os
sys.path.append('/home/karthikeya/svn/repo/lib/pysgpp')
from pysgpp import *
import numpy as np

class FullGrid:
    def __init__(self, dim, levels, basis=None):
        self.dim = dim
        self.levels = levels

        if len(levels) != dim:
            raise Exception('Invalid dimension or level list')
        
        if basis == None:
            self.basis = 'linear'
        else:
            self.basis = basis

        self.size = 0
        for l in levels:
            self.size = self.size + (2**l - 1)

        self.storage = {}
        for i in xrange(self.size):
            grid_point = {}
            for d in xrange(self.dim):
                for j in xrange(2**levels[d] - 1):
                    grid_point[d] = [levels[d], j]
            self.storage[i] = grid_point

    def get_size(self):
        return self.size

    def get_storage(self):
        return self.storage

    def get_levels(self):
        return self.levels

    def get_dim(self):
        return self.dim

    def get_basis(self):
        return self.basis

    def linear_basis_function(self, gpoint, gdim, val):
        prod = 1
        for d in range(gdim):
                lev_ind = gpoint[d]
                level = lev_ind[0]
                index = lev_ind[1]
                basis_map = (2**level)*val[d] - index
                prod = prod * np.maximum(1 - np.absolute(basis_map), 0)
        return prod

    def eval(self, alpha, point):
        result = 0
        for i in xrange(0, self.size):
            result = result + alpha[i] * self.linear_basis_function(self.storage[i], self.dim, point)

        return result

    def evaluate(self, data):
        result = []
        no_cols = data.getNcols()
        no_rows = data.getNrows()
        for j in xrange(no_rows):
                column = []
                val = DataVector(no_cols)
                data.getRow(j, val)
                for i in xrange(0, self.size):
                        column.append(self.linear_basis_function(self.storage[i], self.dim, val))
                result.append(column)
        return DataMatrix(result)
