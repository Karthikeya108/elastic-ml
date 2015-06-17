import sys, os
sys.path.append('/home/karthikeya/svn/repo/lib/pysgpp')
from pysgpp import *
import numpy as np
import itertools

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

        self.size = 1
        for l in levels:
            self.size = self.size * (2**l - 1)

        self.storage = {}
 
        for gp in xrange(self.size):
            self.storage[gp] = {}

        index_dim = []

        for l in levels:
            index_dim.append(2**l)

        ind_list = np.ones(self.dim, dtype=np.int_)
        
        dim_level_index = {}
        for d in xrange(self.dim):
            level_index = []
            for j in xrange(1,2**levels[d]):
                level_index.append([levels[d], j])

            dim_level_index[d] = level_index    

        values = dim_level_index.values()

        level_index_comb = list(itertools.product(*values))

        for g in xrange(self.size):
            dim_lev_ind = {}
            for d in xrange(self.dim):
                dim_lev_ind[d] = level_index_comb[g][d]
            self.storage[g] = dim_lev_ind
        
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

    def func_num_to_index_list(self, point):
        phi = [None] * self.dim
        gpoint = self.storage[point]
        for d in xrange(self.dim):
            lev_ind = gpoint[d]
            index = lev_ind[1]
            phi[d] = index

        return phi

    def get_stiffness_coefficient(self,level,index_1,index_2):
        if (index_1 - index_2) <= 1 and (index_1 - index_2) >= -1:  #phi_1 and phi_2 are neighbors
            if index_1 == index_2:                                  #same points
                if index_1 == 0 or index_1 == 1 << level:           #on boundary
                    return 1 << (level)
                else:                                               #not on boundary
                    return 1 << (level + 1)
            else:                                                   #neighbor point
                return -(1 << level)
        else:
            return 0


    def get_mass_coefficient(self,level,index_1,index_2):
        if (index_1 - index_2) <= 1 and (index_1 - index_2) >= -1:  #phi_1 and phi_2 are neighbors
            if index_1 == index_2:                                 #same points
                if index_1 == 0 or index_1 == 1 << level:          #on boundary
                    return (2.0**-level) / 3.0
                else:                                              #not on boundary
                    return (2.0**-level + 1.0) / 3.0
            else:                                                  #neighbor point
                return (2.0** -level) * (0.5 - 1.0 / 3.0)
        else:
            return 0

    def create_laplacian_matrix(self):
        left_term = 0.0
        right_term = 0.0
        m_collector = 0.0
        function_per_dim = [None] * self.dim

        for d in xrange(self.dim):
            function_per_dim[d] = (1 << self.levels[d]) + 1

        laplacian_matrix = np.ndarray(shape=(self.size,self.size), dtype=float)

        phi_1 = [None] * self.dim
        phi_2 = [None] * self.dim

        for row in xrange(self.size):
            phi_1 = self.func_num_to_index_list(row)    
            for col in xrange(self.size):
                phi_2 = self.func_num_to_index_list(col)
                for d in xrange(self.dim):
                    index_diff = phi_1[d] - phi_2[d]
                    if index_diff >= -1 and index_diff <= 1:
                        s = self.get_stiffness_coefficient(self.levels[d],phi_1[d],phi_2[d])
                        m = self.get_mass_coefficient(self.levels[d],phi_1[d],phi_2[d])
                        
                        if d == 0:
                            left_term = s
                            right_term = m
                            m_collector = m
                        else:
                            left_term = (left_term + right_term) * m
                            right_term = m_collector * s
                            m_collector *= m

                    else:
                        left_term = right_term = m_collector = 0.0

                laplacian_matrix[row,col] = left_term + right_term - m_collector

        return laplacian_matrix
                        
                    

                    
            
