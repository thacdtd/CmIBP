__author__ = 'thac'

import numpy
import scipy.stats
import random


class mIBP:
    def __init__(self):
        # dim_n: number of entities
        # dim_k: number of features
        self.alpha = 1.0
        self.a_m = 1.0

        self.gamma = 1.0
        self.delta = 1.0
        self.b_m = 1.0

        self.matrix_s = numpy.zeros((0, 0))
        self.dim_n = 1
        self.dim_k = 1

    def initialize_data(self, data):
        super

    def