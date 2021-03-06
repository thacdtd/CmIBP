from matplotlib.pylab import *
import numpy
import scipy as sp
import scipy.stats
import random
from scipy.special import erfinv
import math


class LFRM_NEW:
    def __init__(self):
        # dim_n: number of entities
        # dim_k: number of features
        self.alpha = 1.0
        self.beta_w = 1
        self.alpha_hyper_parameter = (1.0, 1.0)
        self.matrix_y = numpy.zeros((0, 0))
        self.matrix_w = numpy.zeros((0, 0))
        self.matrix_z = numpy.zeros((0, 0))
        self.dim_n = 1
        self.dim_k = 1
        self.metropolis_hastings_k_new = False

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
            matrix_z, matrix_w = self.loop_sample_z(i, matrix_z)

        self.dim_k = matrix_z.shape[1]
        print "dim k"
        print self.dim_k
        assert(matrix_z.shape[0] == self.dim_n)
        self.matrix_z = matrix_z.astype(numpy.int)
        self.matrix_w = matrix_w
        return self.matrix_z, self.matrix_w

    def loop_sample_z(self, i_max, matrix_z=None):
        count = 0
        while True:
            count += 1
            new_matrix_z, new_matrix_w = self.sample_new_disk(i_max, matrix_z)
            check = self.check_relational(i_max, new_matrix_z, self.matrix_y, new_matrix_w)
            if check:
                break
            print count
        matrix_z = new_matrix_z
        matrix_w = new_matrix_w
        return matrix_z, matrix_w

    def sample_new_disk(self, i, matrix_z=None):
        if matrix_z is None:
            matrix_z = self.matrix_z

        sample_dish = (numpy.random.uniform(0, 1, (1, matrix_z.shape[1])) < (matrix_z.sum(axis=0).astype(numpy.float) / i))
        dim_k_new = scipy.stats.poisson.rvs((self.alpha * 1.0 / i))

        sample_dish = numpy.hstack((sample_dish, numpy.ones((1, dim_k_new))))

        matrix_z = numpy.hstack((matrix_z, numpy.zeros((matrix_z.shape[0], dim_k_new))))
        matrix_z = numpy.vstack((matrix_z, sample_dish))

        matrix_w = self.sample_w(matrix_z)
        return matrix_z, matrix_w

    def check_relational(self, i_max, matrix_z=None, matrix_y=None, matrix_w=None):
        if matrix_z is None:
            matrix_z = self.matrix_z
        if matrix_y is None:
            matrix_y = self.matrix_y
        if matrix_w is None:
            matrix_w = self.matrix_w

        # i_max = matrix_z.shape[0] - 1
        # matrix_z_new, matrix_w_new = self.sample_new_disk(i_max, matrix_z)

        for i in xrange(0, i_max-1):
            print self.likelihood_y_i_j(i, i_max, matrix_z, matrix_w)
            if self.likelihood_y_i_j(i, i_max, matrix_z, matrix_w) >= 0.5:
                if matrix_y[i][i_max] == 0.0:
                    return False
            else:
                if matrix_y[i][i_max] == 1.0:
                    return False

        return True

    def sample_matrix_z(self):
        order = numpy.random.permutation(self.dim_n)
        for (object_counter, object_index) in enumerate(order):
            # sample Z_n
            singleton_features = self.sample_vector_z_n(object_index)

            if self.metropolis_hastings_k_new:
                # sample K_new using metropolis hasting
                self.sample_metropolis_hastings_k_new(object_index, singleton_features)

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
                prob_z0 = self.log_likelihood_y(self.matrix_z, self.matrix_w, object_index)
                prob_z0 += log_prob_z0[feature_index]
                prob_z0 = numpy.exp(prob_z0)

                # compute the log likelihood when Znk=1
                self.matrix_z[object_index, feature_index] = 1
                prob_z1 = self.log_likelihood_y(self.matrix_z, self.matrix_w, object_index)
                prob_z1 += log_prob_z1[feature_index]
                prob_z1 = numpy.exp(prob_z1)

                z_nk_is_0 = prob_z0 / (prob_z0 + prob_z1)
                if random.random() < z_nk_is_0:
                    self.matrix_z[object_index, feature_index] = 0
                else:
                    self.matrix_z[object_index, feature_index] = 1
        return singleton_features

    def log_likelihood_y(self, matrix_z=None, matrix_w=None, object_index=None):
        if matrix_z is None:
            matrix_z = self.matrix_z
        if matrix_w is None:
            matrix_w = self.matrix_w
        assert(matrix_w.shape == (self.dim_k, self.dim_k))

        log_likelihood = 1
        # compute for each pair
        i = object_index
        z_i = matrix_z[i]
        for j in range(0, self.dim_n-1):
            z_j = matrix_z[j]
            a = 0
            for k in xrange(0, self.dim_k):
                for k_prime in xrange(0, self.dim_k):
                    a += z_i[k] * matrix_w[k][k_prime] * z_j[k_prime]
            a = 1.0/(1.0+numpy.exp(-a))
            log_likelihood *= a
        log_likelihood = numpy.log(log_likelihood)
        return log_likelihood

    def likelihood_y_i_j(self, i=None, j=None, matrix_z=None, matrix_w=None):
        z_i = matrix_z[i]
        z_j = matrix_z[j]
        dim_k = matrix_z.shape[1]

        a = 0
        for k in xrange(0, dim_k):
            for k_prime in xrange(0, dim_k):
                a += z_i[k] * matrix_w[k][k_prime] * z_j[k_prime]
        a = 1.0/(1.0+numpy.exp(-a))
        return a

    def sample_w(self, matrix_z=None):
        if matrix_z is None:
            matrix_z = self.matrix_z

        dim_n = matrix_z.shape[0]
        dim_k = matrix_z.shape[1]
        matrix_w = numpy.zeros((dim_k, dim_k))
        for k in xrange(0, dim_k):
            for k_prime in xrange(k, dim_k):
                matrix_w[k, k_prime] = self.newBetaBinom(self.beta_w, self.beta_w, dim_n,
                                                         self.count_relation_feature(k, k_prime, matrix_z))
                matrix_w[k_prime, k] = matrix_w[k, k_prime]
        #self.matrix_w = matrix_w
        return matrix_w

    def resample_w(self, matrix_z=None):
        # Resample matrix w when add new features
        if matrix_z is None:
            matrix_z = self.matrix_z
        new_matrix_w = self.sample_w(matrix_z)

        return new_matrix_w

    def count_relation_feature(self, k, k_prime, matrix_z=None):
        if matrix_z is None:
            matrix_z = self.matrix_z

        count = 0
        dim_n = matrix_z.shape[0]
        for i in xrange(0, dim_n):
            if (matrix_z[i][k] == matrix_z[i][k_prime]) & (matrix_z[i][k] == 1):
                count += 1
        return count

    def newBetaBinom(self, alpha, beta, n, k):
        a = numpy.random.beta(alpha + k, n - k + beta)
        return self.change_range(a)

    def change_range(self, x):
        a = x*2 - 1
        return a

    def sigmoid(self, x):
        a = 1.0/(1.0+numpy.exp(-x))
        return a

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
        """
        data = numpy.array([[1,0,0],
                            [0,1,1],
                            [0,1,0]])

        self.initialize_data(data)

        matrix_z, matrix_w = self.initialize_matrix_z()
        print matrix_z
        print matrix_w
        count = self.count_relation_feature(0, 1, self.matrix_z)
        print count

        #c = self.log_likelihood_y(self.matrix_z, self.matrix_w, 1)
        #print numpy.exp(c)

        c = self.likelihood_y_i_j(0, 1, self.matrix_z, self.matrix_w)
        print c
        c = self.likelihood_y_i_j(1, 1, self.matrix_z, self.matrix_w)
        print c

        d = self.likelihood_y_i_j(2, 2, self.matrix_z, self.matrix_w)
        print d

        e = self.likelihood_y_i_j(1, 2, self.matrix_z, self.matrix_w)
        print e
        return

if __name__ == '__main__':
    lfrm = LFRM_NEW()
    print lfrm.run()
