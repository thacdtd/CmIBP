__author__ = 'thac'

import numpy
import scipy.stats
import random


class LFRMTimes:
    def __init__(self):
        # dim_n: number of entities
        # dim_k: number of features
        self.alpha = 1.0
        self.sigma_w = 1.0
        self.mean_w = 0
        self.alpha_hyper_parameter = (1.0, 1.0)
        self.matrix_y = numpy.zeros((0, 0))
        self.matrix_w = numpy.zeros((0, 0))
        self.matrix_z = numpy.zeros((0, 0))
        self.dim_n = 1
        self.dim_k = 1
        self.metropolis_hastings_k_new = False

    def initialize_data(self, data):
        self.matrix_y = data
        # print "shape 0"
        # print self.matrix_y.shape[0]
        self.dim_n = self.matrix_y.shape[0]

    """
    initialize latent feature appearance matrix matrix_z according to IBP(alpha)
    """
    def initialize_matrix_z(self):
        matrix_z = numpy.ones((0, 0))
        # initialize matrix matrix_z recursively in IBP manner
        for i in xrange(1, self.dim_n + 1):
            sample_dish = (numpy.random.uniform(0, 1, (1, matrix_z.shape[1])) <
                           (matrix_z.sum(axis=0).astype(numpy.float) / i))

            # sample a value from the poisson distribution, defines the number of new features
            dim_k_new = scipy.stats.poisson.rvs((self.alpha * 1.0 / i))

            sample_dish = numpy.hstack((sample_dish, numpy.ones((1, dim_k_new))))

            matrix_z = numpy.hstack((matrix_z, numpy.zeros((matrix_z.shape[0], dim_k_new))))
            matrix_z = numpy.vstack((matrix_z, sample_dish))

        self.dim_k = matrix_z.shape[1]
        assert(matrix_z.shape[0] == self.dim_n)
        self.matrix_z = matrix_z.astype(numpy.int)
        self.matrix_w = numpy.random.normal(0, self.sigma_w, (self.dim_k, self.dim_k))
        return self.matrix_z
