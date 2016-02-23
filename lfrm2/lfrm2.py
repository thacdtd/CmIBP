from matplotlib.pylab import *
import numpy
import scipy as sp
import scipy.stats
import random
from scipy.special import erfinv
import math


class LFRM2:
    def __init__(self):
        # dim_n: number of entities
        # dim_k: number of features
        self.alpha = 1.0
        self.sigma_w = 1.0
        self.alpha_hyper_parameter = (1.0, 1.0)
        self.matrix_y = numpy.zeros((0, 0))
        self.matrix_w = numpy.zeros((0, 0))
        self.matrix_z = numpy.zeros((0, 0))
        self.dim_n = 1
        self.dim_k = 1
        self.metropolis_hastings_k_new = False

        self.amount_fix = 0.01

    def initialize_data(self, data):
        self.matrix_y = data
        self.dim_n = self.matrix_y.shape[0]

    """
    initialize latent feature appearance matrix matrix_z according to IBP(alpha)
    """
    def initialize_matrix_z(self):
        matrix_z = numpy.ones((0, 0))
        # initialize matrix matrix_z recursively in IBP manner
        sample_dish = [[]]
        dim_k_new = scipy.stats.poisson.rvs((self.alpha * 1.0))

        sample_dish = numpy.hstack((sample_dish, numpy.ones((1, dim_k_new))))

        matrix_z = numpy.hstack((matrix_z, numpy.zeros((matrix_z.shape[0], dim_k_new))))
        matrix_z = numpy.vstack((matrix_z, sample_dish))
        for i in xrange(1, self.dim_n):
            matrix_z = self.sample_new_disk(i, matrix_z)

        self.dim_k = matrix_z.shape[1]
        assert(matrix_z.shape[0] == self.dim_n)
        self.matrix_z = matrix_z.astype(numpy.int)

        self.matrix_w = self.sample_w()
        return self.matrix_z

    def sample_new_disk(self, i, matrix_z=None):
        if matrix_z is None:
            matrix_z = self.matrix_z

        sample_dish = (numpy.random.uniform(0, 1, (1, matrix_z.shape[1])) < (matrix_z.sum(axis=0).astype(numpy.float) / i))
        dim_k_new = scipy.stats.poisson.rvs((self.alpha * 1.0 / i))

        sample_dish = numpy.hstack((sample_dish, numpy.ones((1, dim_k_new))))

        matrix_z = numpy.hstack((matrix_z, numpy.zeros((matrix_z.shape[0], dim_k_new))))
        matrix_z = numpy.vstack((matrix_z, sample_dish))

        return matrix_z

    def log_likelihood_y(self, matrix_z=None, matrix_w=None, matrix_y=None):
        if matrix_z is None:
            matrix_z = self.matrix_z
        if matrix_w is None:
            matrix_w = self.matrix_w
        if matrix_y is None:
            matrix_y = self.matrix_y
        assert(matrix_w.shape == (self.dim_k, self.dim_k))

        log_likelihood = 1
        dim_n = self.dim_n
        for i in range(0, dim_n-1):
            for j in range(0, dim_n-1):
                log_likelihood *= self.likelihood_y_i_j(i, j, matrix_z, matrix_w, matrix_y)

        log_likelihood = numpy.log(log_likelihood)
        return log_likelihood

    def likelihood_y_i_j(self, i=0, j=0, matrix_z=None, matrix_w=None, matrix_y=None):
        if matrix_z is None:
            matrix_z = self.matrix_z
        if matrix_w is None:
            matrix_w = self.matrix_w
        if matrix_y is None:
            matrix_y = self.matrix_y

        dim_k = self. dim_k
        likelihood_old = numpy.dot(numpy.dot(matrix_z[i], matrix_w), matrix_z[j].transpose())

        likelihood_new = 0
        matrix_w_new = matrix_w
        for k in range(0, dim_k-1):
            for k_prime in range(0, dim_k-1):
                if (matrix_z[i][k] == 1) and (matrix_z[j][k_prime] == 1):
                    # metropolis hasting fix weight w[k][k_prime]
                    amount_fix = numpy.random.uniform(0, 0.1)
                    if matrix_y[i][j] == 1:
                        matrix_w_new[k][k_prime] += amount_fix
                    else:
                        matrix_w_new[k][k_prime] -= amount_fix
                    likelihood_new += matrix_w_new[k][k_prime]

        accept_new = min([1., likelihood_new/likelihood_old])
        if numpy.random.uniform(0, 1) < accept_new:
            self.matrix_w = matrix_w_new
            likelihood = likelihood_new
        else:
            likelihood = likelihood_old

        likelihood = 1.0/(1.0+numpy.exp(-likelihood))

        if matrix_y[i][j] == 0:
            likelihood = 1 - likelihood

        if matrix_y[i][j] == -1:
            if likelihood >= 0.5:
                self.matrix_y[i][j] = 1
            else:
                self.matrix_y[i][j] = 0
        return likelihood

    def sdnorm(z):
        return numpy.exp(-z*z/2.)/numpy.sqrt(2*numpy.pi)

    def sample_vector_z_n(self, object_index):
        assert(type(object_index) == int or type(object_index) == numpy.int32 or type(object_index) == numpy.int64)

        # calculate initial feature possess counts
        m = self.matrix_z.sum(axis=0)

        # remove this data point from m vector
        new_m = (m - self.matrix_z[object_index, :]).astype(numpy.float)

        # compute the log probability of p(Znk=0 | Z_nk) and p(Znk=1 | Z_nk)
        log_prob_z1 = numpy.log(new_m / self.dim_n)
        log_prob_z0 = numpy.log(1.0 - (new_m / self.dim_n))

        # find all singleton features possessed by current object
        singleton_features = [nk for nk in range(self.dim_k) if self.matrix_z[object_index, nk] != 0 and new_m[nk] == 0]
        non_singleton_features = [nk for nk in range(self.dim_k) if nk not in singleton_features]

        order = numpy.random.permutation(self.dim_k)

        for (feature_counter, feature_index) in enumerate(order):
            if feature_index in non_singleton_features:

                # compute the log likelihood when Znk=0
                self.matrix_z[object_index, feature_index] = 0
                prob_z0 = self.log_likelihood_y(self.matrix_z, self.matrix_w, self.matrix_y)
                prob_z0 += log_prob_z0[feature_index]
                prob_z0 = numpy.exp(prob_z0)

                # compute the log likelihood when Znk=1
                self.matrix_z[object_index, feature_index] = 1
                prob_z1 = self.log_likelihood_y(self.matrix_z, self.matrix_w, self.matrix_y)
                prob_z1 += log_prob_z1[feature_index]
                prob_z1 = numpy.exp(prob_z1)

                z_nk_is_0 = prob_z0 / (prob_z0 + prob_z1)
                if random.random() < z_nk_is_0:
                    self.matrix_z[object_index, feature_index] = 0
                else:
                    self.matrix_z[object_index, feature_index] = 1
        return singleton_features

    def sample_matrix_z(self):
        order = numpy.random.permutation(self.dim_n)
        for (object_counter, object_index) in enumerate(order):
            # sample Z_n
            singleton_features = self.sample_vector_z_n(object_index)

            if self.metropolis_hastings_k_new:
                # sample K_new using metropolis hasting
                #b = self.sample_metropolis_hastings_k_new(object_index, singleton_features)
                print "metropolis hasting"

    def sample(self, iterations):
        vec = []
        for i in range(0, self.dim_k):
            vec.append([])

        for iter in xrange(iterations):
            self.sample_matrix_z()

            #self.regularize_matrices()
            #self.alpha = self.sample_alpha()
            # print iter, self.dim_k
            # print("alpha: %f\tsigma_w: %f\tmean_w: %f" % (self.alpha, self.sigma_w, self.mean_w))
            print self.matrix_z.sum(axis=0)
            aa = self.matrix_z.sum(axis=0)
            for i in range(0, self.dim_k):
                vec[i].append(aa[i])
        return vec

    def sample_w(self):
        matrix_w = numpy.random.normal(0, self.sigma_w, (self.dim_k, self.dim_k))
        return matrix_w

    def run(self):
        """
        data = numpy.array([[0,0,1,0,1,0,0,1,0],
                            [0,0,0,0,0,0,1,0,1],
                            [0,0,1,0,0,0,1,0,1],
                            [0,1,1,0,0,0,0,1,0],
                            [0,0,0,0,0,0,1,0,1],
                            [0,1,1,0,1,0,0,1,0],
                            [1,0,0,0,0,1,0,0,0],
                            [0,0,0,0,0,1,1,0,1],
                            [1,0,0,1,0,1,0,0,0]])

        data = numpy.array([[1,0,0,1,0],
                            [0,1,1,1,0],
                            [0,1,0,1,1],
                            [1,0,0,1,1],
                            [1,1,0,1,1]])
        """

        data = numpy.array([[1,1,1,1,0,0,0,0,0,0,0,0,0,0],
                            [1,1,1,1,0,0,0,0,0,0,0,0,0,0],
                            [1,1,1,1,0,0,0,0,0,0,0,0,0,0],
                            [1,1,1,1,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,1,1,1,1,0,0,0,0,0,0],
                            [0,0,0,0,1,1,1,1,0,0,0,0,0,0],
                            [0,0,0,0,1,1,1,1,0,0,0,0,0,0],
                            [0,0,0,0,1,1,1,1,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,1,1,1,1,1],
                            [0,0,0,0,0,0,0,0,1,1,1,1,1,1],
                            [0,0,0,0,0,0,0,0,1,1,1,1,1,1],
                            [0,0,0,0,0,0,0,0,1,1,1,1,1,1],
                            [0,0,0,0,0,0,0,0,1,1,1,1,1,1],
                            [0,0,0,0,0,0,0,0,1,1,1,1,1,1]])

        self.initialize_data(data)

        matrix_z = self.initialize_matrix_z()
        #matrix_w = self.sample_w()
        print matrix_z
        #print matrix_w

        self.sample(100)
        print self.matrix_z
        print self.matrix_w
if __name__ == '__main__':
    lfrm = LFRM2()
    print lfrm.run()
