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
        self.dim_m = 1
        self.metropolis_hastings_k_new = False

    def initialize_data(self, data):
        self.matrix_y = data
        # print "shape 0"
        # print self.matrix_y.shape[0]
        self.dim_n = self.matrix_y.shape[0]
        self.dim_m = self.matrix_y.shape[2]

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

    def log_likelihood_y(self, matrix_y=None, matrix_z=None, matrix_w=None):
        # if matrix_y is None:
            # matrix_y = self.matrix_y
        if matrix_z is None:
            matrix_z = self.matrix_z
        if matrix_w is None:
            matrix_w = self.matrix_w

        assert(matrix_w.shape == (self.dim_k, self.dim_k))

        log_likelihood = numpy.dot(numpy.dot(matrix_z, matrix_w), matrix_z.T)

        log_likelihood = numpy.log(1.0/(1.0+numpy.exp(-log_likelihood)))
        return log_likelihood

    """
    @param object_index: an int data type, indicates the object index n (row index) of Z we want to sample
    """
    def sample_vector_z_m_n(self, object_index, previous_matrix_z_m):
        if not previous_matrix_z_m:
            print "no previous info"
        assert(type(object_index) == int or type(object_index) == numpy.int32 or type(object_index) == numpy.int64)

        # calculate initial feature possess counts
        m = self.matrix_z.sum(axis=0)

        # remove this data point from m vector
        new_m = (m - self.matrix_z[object_index, :]).astype(numpy.float)

        # compute the log probability of p(Znk=0 | Z_nk) and p(Znk=1 | Z_nk)
        log_prob_z1 = numpy.log(new_m * (new_m / self.dim_n))
        log_prob_z0 = numpy.log(new_m * (1.0 - new_m / self.dim_n))

        # find all singleton features possessed by current object
        singleton_features = [nk for nk in range(self.dim_k) if self.matrix_z[object_index, nk] != 0 and new_m[nk] == 0]
        non_singleton_features = [nk for nk in range(self.dim_k) if nk not in singleton_features]

        order = numpy.random.permutation(self.dim_k)

        for (feature_counter, feature_index) in enumerate(order):
            if feature_index in non_singleton_features:

                # compute the log likelihood when Znk=0
                self.matrix_z[object_index, feature_index] = 0
                prob_z0 = self.log_likelihood_y(self.matrix_y[[object_index], :], self.matrix_z[[object_index], :],
                                                self.matrix_w)
                prob_z0 += log_prob_z0[feature_index]
                prob_z0 = numpy.exp(prob_z0)

                # compute the log likelihood when Znk=1
                self.matrix_z[object_index, feature_index] = 1
                prob_z1 = self.log_likelihood_y(self.matrix_y[[object_index], :], self.matrix_z[[object_index], :],
                                                self.matrix_w)
                prob_z1 += log_prob_z1[feature_index]
                prob_z1 = numpy.exp(prob_z1)

                z_nk_is_0 = prob_z0 / (prob_z0 + prob_z1)
                if random.random() < z_nk_is_0:
                    self.matrix_z[object_index, feature_index] = 0
                else:
                    self.matrix_z[object_index, feature_index] = 1
        return singleton_features

    def sample_matrix_z_m(self, previous_matrix_z_m):
        order = numpy.random.permutation(self.dim_n)
        for (object_counter, object_index) in enumerate(order):
            # sample Z_n
            singleton_features = self.sample_vector_z_m_n(object_index, previous_matrix_z_m)

            if self.metropolis_hastings_k_new:
                # sample K_new using metropolis hasting
                b = self.sample_metropolis_hastings_k_new(object_index, singleton_features)
                # print b

    def sample_matrix_z(self):
        for i in range(0, self.dim_m):
            print i

    def load_data(self, file_location):
        import scipy.io
        mat_vals = scipy.io.loadmat(file_location)
        datas = mat_vals['datas']
        data_num = mat_vals['dataNum']
        t_time = mat_vals['tTime']
        return datas, data_num, t_time

    def run(self):
        print "lfrm_mibp"
        datas, data_num, t_time = self.load_data('../data/enrondata.mat')
        self.initialize_data(datas)
        self.sample_matrix_z()
        print datas.shape

if __name__ == '__main__':
    lfrmtimes = LFRMTimes()
    #lfrm_mibp.sample_a_m()
    lfrmtimes.run()
