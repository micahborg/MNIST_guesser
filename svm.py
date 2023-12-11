#
# Code from https://github.com/AbirHaque/MATH582KernelGroup/blob/main/svm.py
#

import numpy as np
import pandas as pd
import rbf_kernel as rbf
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False

def calculate_kernel(self, X, y, C, sigma):
        return C * np.exp( -1 * ( np.linalg.norm( X - y ) ** 2 ) / ( 2 * sigma ** 2 ) )

class SVM:

    def __init__( self, C = 0.1 ):
        self.C = C
        
    def fit(self, X: np.array, y: np.array):
        Y = y.to_numpy()
        N, m = X.shape

        K = np.zeros( ( N,N ) )
        for i in range( N ):
            for j in range( N ):
                K[ i, j ] = calculate_kernel( X = X[ i, : ], y = X[ j, : ], sigma = 1 )

        P = matrix( Y @ K @ Y ) # check for small eiganvalue for kernel optimization
        q = matrix( np.ones( ( N,1 ) ) * -1 ) # negative ones
        G = matrix( np.vstack( ( y.T,
                              -1 * y.T,
                              -1 * np.eye( N ),
                              np.eye( N ) ) ) ) # stack the matrices vertically
        h = matrix( np.vstack( ( np.zeros( ( N + 2, 1 ) ),
                              self.C * np.ones( ( N, 1 ) ) ) ) )
        
        while True:
            try:
                solution = solvers.qp( P, q, G, h )
                break
            
            except:
                pass

        alphas = np.array( solution[ 'x' ] )

        self.w = np.dot( ( alphas * y ).T, X )
        self.b = np.median( np.array( [ np.abs( y[ n, : ] - calculate_kernel( X = self.w, y = X[ n, : ], C = self.C, sigma = 1 ) ) for n in range( N ) ] ) )

    def predict( self, X: pd.DataFrame ):
        N = X.shape[ 0 ]
        output = X @ self.w.T + np.ones( ( N, 1 ) ) * self.b
        for i in range( len( output ) ):
            if output[ i ][ 0 ] < 0:
                output[ i ][ 0 ] = -1

            if output[ i ][ 0 ] > 0:
                output[ i ][ 0 ] = 1

        return output