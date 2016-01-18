__author__ = 'thac'

import numpy
import scipy.stats
import random


class LFRMmIBP:
    def __init__(self):
        # dim_n: number of entities
        # dim_k: number of features
        self.alpha = 1.0
        self.a_m = 1.0

        self.gamma = 1.0
        self.delta = 1.0
        self.b_m = 1.0

        self.matrix_s = numpy.zeros((0, 0))
        self.matrix_y = numpy.zeros((0, 0))
        self.matrix_w = numpy.zeros((0, 0))

        self.dim_n = 5
        self.dim_k = 1

    def initialize_data(self, data):
        self.matrix_y = data
        self.dim_n = self.matrix_y.shape[0]

    def initialize_matrix_s(self):
        matrix_s = numpy.ones((0, 0))
        # initialize matrix matrix_s recursively in IBP manner
        for i in xrange(1, self.dim_n + 1):
            sample_dish = (numpy.random.uniform(0, 1, (1, matrix_s.shape[1])) <
                           (matrix_s.sum(axis=0).astype(numpy.float) / i))

            # sample a value from the poisson distribution, defines the number of new features
            dim_k_new = scipy.stats.poisson.rvs((self.alpha * 1.0 / i))

            sample_dish = numpy.hstack((sample_dish, numpy.ones((1, dim_k_new))))

            matrix_s = numpy.hstack((matrix_s, numpy.zeros((matrix_s.shape[0], dim_k_new))))
            matrix_s = numpy.vstack((matrix_s, sample_dish))

        self.dim_k = matrix_s.shape[1]
        assert(matrix_s.shape[0] == self.dim_n)
        self.matrix_s = matrix_s.astype(numpy.int)
        #self.matrix_w = numpy.random.normal(0, self.sigma_w, (self.dim_k, self.dim_k))
        return self.matrix_s

    def sample_vector_s_n(self, object_index):
        assert(type(object_index) == int or type(object_index) == numpy.int32 or type(object_index) == numpy.int64)

        # calculate initial feature possess counts
        m = self.matrix_s.sum(axis=0)

        # remove this data point from m vector
        new_m = (m - self.matrix_s[object_index, :]).astype(numpy.float)

        # compute the log probability of p(Znk=0 | Z_nk) and p(Znk=1 | Z_nk)
        log_prob_z1 = numpy.log(new_m * (new_m / self.dim_n))
        log_prob_z0 = numpy.log(new_m * (1.0 - new_m / self.dim_n))

        # find all singleton features possessed by current object
        singleton_features = [nk for nk in range(self.dim_k) if self.matrix_s[object_index, nk] != 0 and new_m[nk] == 0]
        non_singleton_features = [nk for nk in range(self.dim_k) if nk not in singleton_features]

        order = numpy.random.permutation(self.dim_k)

        for (feature_counter, feature_index) in enumerate(order):
            if feature_index in non_singleton_features:

                # compute the log likelihood when Znk=0
                self.matrix_s[object_index, feature_index] = 0
                prob_z0 = self.log_likelihood_y(self.matrix_y[[object_index], :], self.matrix_s[[object_index], :],
                                                self.matrix_w)
                prob_z0 += log_prob_z0[feature_index]
                prob_z0 = numpy.exp(prob_z0)

                # compute the log likelihood when Znk=1
                self.matrix_s[object_index, feature_index] = 1
                prob_z1 = self.log_likelihood_y(self.matrix_y[[object_index], :], self.matrix_s[[object_index], :],
                                                self.matrix_w)
                prob_z1 += log_prob_z1[feature_index]
                prob_z1 = numpy.exp(prob_z1)

                z_nk_is_0 = prob_z0 / (prob_z0 + prob_z1)
                if random.random() < z_nk_is_0:
                    self.matrix_s[object_index, feature_index] = 0
                else:
                    self.matrix_s[object_index, feature_index] = 1
        return singleton_features
    
    def sample_a_m(self):
        a_1 = numpy.random.beta(self.alpha, 1)
        a_2 = numpy.random.beta(self.alpha, 1)
        print a_1
        print a_2
        a_2 = self.alpha*pow(a_1, -self.alpha)*pow(a_2, (self.alpha - 1))
        print a_1
        print a_2

    def run(self):
        print "lfrm_mibp"

        matrix_s = self.initialize_matrix_s()
        print matrix_s

if __name__ == '__main__':
    lfrm_mibp = LFRMmIBP()
    lfrm_mibp.sample_a_m()
    #lfrm_mibp.run()
