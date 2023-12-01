import numpy as np

class rbf_kernel:
    def __init__(self):
        pass

    def calculate_kernel(self, X, y, C, sigma):
        return C * np.exp( -1 * ( np.linalg.norm( X - y ) ** 2 ) / ( 2 * sigma ** 2 ) )