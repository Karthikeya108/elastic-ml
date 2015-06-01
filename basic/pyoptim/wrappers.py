'''
Created on Feb 11, 2015

@author: perun
'''
import sys
sys.path.append('/home/karthikeya/svn/repo/lib/pysgpp')
from pysgpp import Grid, DataVector, DataMatrix, createOperationMultipleEval
from pysgpp import createOperationLTwoDotProduct
from algorithms import vSGD, SGD, vSGDfd
from core.datainterface import DatasetWrapper, ModuleWrapper
import numpy as np

class SparseGridWrapper(DatasetWrapper):
    '''
    Regression with adaptive sparse grids using stochastic gradient descent 
    '''


    def __init__(self, grid, dataset, l):
        '''
        Constructor
        '''
        self.grid = grid
        self.dim = grid.get_dim()
        self.dataset = dataset
        self.l = l
        self._ready = False
        self.reset(dataset)
        
    
    def loss_fun(self, params):
        '''
        Compute the value of regularized quadratic loss function in the current state
        '''
        if not hasattr(self, '_lastseen'):
            return np.inf
        else:
            params_DV = DataVector(params)
            residuals = []
            for sample_idx in xrange(self.batch_size):
                x = self._lastseen[sample_idx, :self.dim]
                y = self._lastseen[sample_idx, self.dim]
                x_DV = DataVector(x)
                
                residual = self.grid.eval(params_DV, x_DV) - y
                residuals.append(residual*residual)
            regularizer = params_DV.l2Norm()
            regularizer = self.l*regularizer*regularizer
            return 0.5*np.mean(residuals) + regularizer
    
    
    def gradient_fun(self, params):
        '''
        Compute the gradient vector in the current state
        '''
        #import ipdb; ipdb.set_trace() #
        gradient_array = np.empty((self.batch_size, self.grid.get_size()))
        for sample_idx in xrange(self.batch_size):
            x = self._lastseen[sample_idx, :self.dim]
            y = self._lastseen[sample_idx, self.dim]
            params_DV = DataVector(params)
            
            gradient = DataVector(len(params_DV))
            
            single_alpha = DataVector(1)
            single_alpha[0] = 1
            
            data_matrix = DataMatrix(x.reshape(1,-1))
        
            mult_eval = self.grid.evaluate(data_matrix)
            #mult_eval.multTranspose(single_alpha, gradient)
            mult_eval.transpose()
            gradient = DataVector(np.dot(mult_eval.array(), single_alpha.array()))
         
            residual = gradient.dotProduct(params_DV) - y;
            gradient.mult(residual);
            #import ipdb; ipdb.set_trace() #

            gradient_array[sample_idx, :] =  gradient.array()
        return gradient_array
    
    
    def _provide(self):
        self._lastseen = np.ndarray(self.batch_size*(self.dim+1)).reshape(self.batch_size, -1)
        for idx in xrange(self.batch_size):
            self._lastseen[idx,:] = self.dataset[self.getIndex(),:]
            self._counter += 1
            
            
    def currentDiagHess(self, params):
        #return np.ones(params.shape)
#         if hasattr(self, 'H'):
#             return self.H
#         op_l2_dot = createOperationLTwoDotProduct(self.grid)
#         self.H = np.empty((self.grid.getSize(), self.grid.getSize()))
#         u = DataVector(self.grid.getSize())
#         u.setAll(0.0)
#         result = DataVector(self.grid.getSize())
#         for grid_idx in xrange(self.grid.getSize()):
#             u[grid_idx] = 1.0
#             op_l2_dot.mult(u, result)
#             self.H[grid_idx,:] = result.array()
#             u[grid_idx] = 0.0
#         self.H = np.diag(self.H).reshape(1,-1)
#         return self.H
        #import ipdb; ipdb.set_trace()
        size = self._lastseen.shape[0]
        data_matrix = DataMatrix(self._lastseen[:,:self.dim])
        #mult_eval = createOperationMultipleEval(self.grid, data_matrix)
        mult_eval = self.grid.evaluate(data_matrix)

        params_DV = DataVector(self.grid.get_size())
        params_DV.setAll(0.)
        results_DV = DataVector(size)
        self.H = np.zeros(self.grid.get_size())
        for i in xrange(self.grid.get_size()):
            params_DV[i] = 1.0
            #mult_eval.mult(params_DV, results_DV)
            results_DV = DataVector(np.dot(mult_eval.array(),params_DV.array()))
            self.H[i] = results_DV.l2Norm()**2
            params_DV[i] = 0.0
        self.H = self.H.reshape(1,-1)/size
        #import ipdb; ipdb.set_trace() 
        return self.H
        


def printy(s):
    if s._num_updates % 2 == 0:
        print s._num_updates, s.bestParameters, 
        print np.mean(s.provider.currentLosses(s.bestParameters))


if __name__ == '__main__':
    dim = 2
    level = 3
    l = 1E-8
    grid = Grid.createLinearGrid(dim)
    gridStorage = grid.getStorage()
    gridGen = grid.createGridGenerator()
    gridGen.regular(level)
    
    size = 10
    x = np.random.rand(size*dim).reshape(size,-1)
    y = np.apply_along_axis(lambda p:4*p[0]*(1-p[0])*p[1]*(1-p[1]), 1, x)
    dataset = np.vstack([x.T,y]).T
     
    f = SparseGridWrapper(grid, dataset, l)
    x0 = np.ones(grid.getSize())
#     #algo = SGD(f, x0, callback=printy, learning_rate=0.2, loss_target=-np.inf)
#     algo = vSGDfd(f, x0, loss_target=-np.inf, init_samples=100)
#     algo.run(10000)
#     printy(algo)
    import matplotlib
    import seaborn as sns
    sns.set()
    H = f.currentDiagHess(x0)
    sns.heatmap(H, annot=True)
    print "done"
