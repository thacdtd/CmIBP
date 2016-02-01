__author__ = 'thac'

from matplotlib.pylab import *
import numpy
import scipy.stats
import random
import rtnorm
from scipy.special import erfinv
import math


class LFRM:
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
        self.matrix_w = numpy.exp(self.log_likelihood_w(self.matrix_y, self.matrix_z)) #numpy.random.normal(0, self.sigma_w, (self.dim_k, self.dim_k))
        self.mean_w = numpy.mean(self.matrix_w)
        self.sigma_w = numpy.std(self.matrix_w)
        print scipy.stats.norm.cdf(self.mean_w/numpy.sqrt(2.892 + self.sigma_w**2))
        print self.matrix_w
        return self.matrix_z

    def log_likelihood_y(self, matrix_z=None, matrix_w=None):
        if matrix_z is None:
            matrix_z = self.matrix_z
        if matrix_w is None:
            matrix_w = self.matrix_w

        assert(matrix_w.shape == (self.dim_k, self.dim_k))

        log_likelihood = numpy.dot(numpy.dot(matrix_z, matrix_w), matrix_z.T)

        log_likelihood = scipy.stats.norm.cdf(log_likelihood) #numpy.log(1.0/(1.0+numpy.exp(-log_likelihood)))

        return log_likelihood

    """
    @param object_index: an int data type, indicates the object index n (row index) of Z we want to sample
    """
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
                prob_z0 = self.log_likelihood_y(self.matrix_z[object_index, :], self.matrix_w)
                prob_z0 += log_prob_z0[feature_index]
                prob_z0 = numpy.exp(prob_z0)

                # compute the log likelihood when Znk=1
                self.matrix_z[object_index, feature_index] = 1
                prob_z1 = self.log_likelihood_y(self.matrix_z[object_index, :], self.matrix_w)
                prob_z1 += log_prob_z1[feature_index]
                prob_z1 = numpy.exp(prob_z1)

                z_nk_is_0 = prob_z0 / (prob_z0 + prob_z1)
                if random.random() < z_nk_is_0:
                    self.matrix_z[object_index, feature_index] = 0
                else:
                    self.matrix_z[object_index, feature_index] = 1
        return singleton_features

    """
    sample K_new using metropolis hastings algorithm
    """
    def sample_metropolis_hastings_k_new(self, object_index, singleton_features):
        if type(object_index) != list:
            object_index = [object_index]

        k_temp = scipy.stats.poisson.rvs(self.alpha / self.dim_n)

        if k_temp <= 0 and len(singleton_features) <= 0:
            return False

        # generate new features from a normal distribution with mean 0 and variance sigma_a, a K_new-by-K matrix

        matrix_w_new = self.matrix_w
        matrix_w_new = matrix_w_new[[k for k in xrange(self.dim_k) if k not in singleton_features], :]
        matrix_w_new = matrix_w_new[:, [k for k in xrange(self.dim_k) if k not in singleton_features]]
        if k_temp != 0:
            for i in range(0, k_temp, 1):
                matrix_w_temp = numpy.random.normal(0, self.sigma_w, (1, matrix_w_new.shape[0]))
                matrix_w_new = numpy.vstack((matrix_w_new, matrix_w_temp))
                matrix_w_temp = numpy.hstack((matrix_w_temp, [[0]]))
                matrix_w_new = numpy.hstack((matrix_w_new, matrix_w_temp.T))

        # generate new z matrix row
        matrix_z_new = numpy.hstack((self.matrix_z[[object_index], [k for k in xrange(self.dim_k)
                                    if k not in singleton_features]], numpy.ones((len(object_index), k_temp))))

        dim_k_new = self.dim_k + k_temp - len(singleton_features)

        # compute the probability of generating new features
        prob_new = numpy.exp(self.log_likelihood_y(self.matrix_y[object_index, :], matrix_z_new, matrix_w_new))

        # construct the A_old and Z_old
        matrix_w_old = self.matrix_w
        vector_z_old = self.matrix_z[object_index, :]
        dim_k_old = self.dim_k

        assert(matrix_w_old.shape == (dim_k_old, dim_k_old))
        assert(matrix_w_new.shape == (dim_k_new, dim_k_new))
        assert(vector_z_old.shape == (len(object_index), dim_k_old))
        assert(matrix_z_new.shape == (len(object_index), dim_k_new))

        # compute the probability of using old features
        prob_old = numpy.exp(self.log_likelihood_y(self.matrix_y[object_index, :], vector_z_old, matrix_w_old))

        # compute the probability of generating new features
        prob_new /= (prob_old + prob_new)

        # if we accept the proposal, we will replace old W and Z matrices
        if random.random() < prob_new:
            self.matrix_w = matrix_w_new
            self.matrix_z = numpy.hstack((self.matrix_z[:, [k for k in xrange(self.dim_k)
                                                            if k not in singleton_features]],
                                          numpy.zeros((self.dim_n, k_temp))))
            self.matrix_z[object_index, :] = matrix_z_new
            self.dim_k = dim_k_new
            return True

        return False

    def sample_matrix_z(self):
        order = numpy.random.permutation(self.dim_n)
        for (object_counter, object_index) in enumerate(order):
            # sample Z_n
            singleton_features = self.sample_vector_z_n(object_index)

            if self.metropolis_hastings_k_new:
                # sample K_new using metropolis hasting
                b = self.sample_metropolis_hastings_k_new(object_index, singleton_features)
                # print b

    def sample_matrix_w(self, old_matrix_z, new_matrix_z):
        # construct the W_old
        #matrix_w_old = self.matrix_w
        #innov = numpy.random.uniform(-self.sigma_w + self.mean_w, self.sigma_w + self.mean_w, (self.dim_k, self.dim_k))
        #matrix_w_new = matrix_w_old + innov
        log_prob_new = self.log_likelihood_w(self.matrix_y, new_matrix_z)

        log_prob_old = numpy.log(self.matrix_w)

        # compute the probability of generating new features
        #log_prob_new /= (log_prob_old + log_prob_new)

        #aprob = min([1., prob_new.all()/prob_old.all()])  # acceptance probability
        #u = numpy.random.uniform(0, 1)
        #if u < aprob:

        mean_new = numpy.mean(numpy.exp(log_prob_new))
        std_new = numpy.std(numpy.exp(log_prob_new))
        mean_old = numpy.mean(numpy.exp(log_prob_old))
        std_old = numpy.std(numpy.exp(log_prob_old))

        cdf_new = scipy.stats.norm.cdf(mean_new/numpy.sqrt(2.892 + std_new**2))
        cdf_old = scipy.stats.norm.cdf(mean_old/numpy.sqrt(2.892 + std_old**2))

        aprob = min([1., cdf_new/cdf_old])
        # if we accept the proposal, we will replace old W matrices
        u = numpy.random.uniform(0, 1)
        if u < aprob:#prob_new.all():
            # construct W_new
            # self.matrix_w = numpy.exp(log_prob_new) # matrix_w_new
            # print log_prob_new
            self.mean_w = numpy.mean(numpy.exp(log_prob_new))
            self.sigma_w = numpy.std(numpy.exp(log_prob_new))
            self.matrix_w = numpy.exp(log_prob_new) #numpy.random.normal(0, self.sigma_w, (self.dim_k, self.dim_k))
            return True
        return False

    def log_likelihood_w(self, matrix_y=None, matrix_z=None):
        if matrix_y is None:
            matrix_y = self.matrix_y
        if matrix_z is None:
            matrix_z = self.matrix_z

        assert(matrix_y.shape == (self.dim_n, self.dim_n))

        log_likelihood = numpy.dot(numpy.dot(matrix_z.T, matrix_y), matrix_z)

        log_likelihood = scipy.stats.norm.cdf(log_likelihood)
        #log_likelihood = numpy.log(1.0/(1.0+numpy.exp(-log_likelihood)))
        return log_likelihood
    """
    remove the empty column in matrix Z and the corresponding feature in W
    """
    def regularize_matrices(self):
        assert(self.matrix_z.shape == (self.dim_n, self.dim_k))
        z_sum = numpy.sum(self.matrix_z, axis=0)
        assert(len(z_sum) == self.dim_k)
        indices = numpy.nonzero(z_sum == 0)

        self.matrix_z = self.matrix_z[:, [k for k in range(self.dim_k) if k not in indices]]
        self.matrix_w = self.matrix_w[[k for k in range(self.dim_k) if k not in indices], :]

        self.dim_k = self.matrix_z.shape[1]
        assert(self.matrix_z.shape == (self.dim_n, self.dim_k))
        assert(self.matrix_w.shape == (self.dim_k, self.dim_k))

    """
    sample alpha from conjugate posterior
    """
    def sample_alpha(self):
        assert(self.alpha_hyper_parameter is not None)
        assert(type(self.alpha_hyper_parameter) == tuple)

        (alpha_hyper_a, alpha_hyper_b) = self.alpha_hyper_parameter

        posterior_shape = alpha_hyper_a + self.dim_k
        h_n = numpy.array([range(self.dim_n)]) + 1.0
        h_n = numpy.sum(1.0 / h_n)
        posterior_scale = 1.0 / (alpha_hyper_b + h_n)

        alpha_new = scipy.stats.gamma.rvs(posterior_shape, scale=posterior_scale)
        return alpha_new

    def sample(self, iterations):
        vec = []
        temp = []
        for i in range(0, self.dim_k):
            vec.append([])

        for iter in xrange(iterations):
            old_matrix_z = self.matrix_z
            self.sample_matrix_z()
            new_matrix_z = self.matrix_z
            self.regularize_matrices()
            # self.sample_matrix_w(old_matrix_z, new_matrix_z)
            #self.matrix_w = numpy.exp(self.log_likelihood_w(self.matrix_y, self.matrix_z))
            self.alpha = self.sample_alpha()
            # print iter, self.dim_k
            # print("alpha: %f\tsigma_w: %f\tmean_w: %f" % (self.alpha, self.sigma_w, self.mean_w))
            aa = self.matrix_z.sum(axis=0)
            for i in range(0, self.dim_k):
                vec[i].append(aa[i])
            #print self.matrix_w
        return vec

    def load_data(self, file_location):
        import scipy.io
        mat_vals = scipy.io.loadmat(file_location)
        print mat_vals
        datas = mat_vals['datas']
        data_num = mat_vals['dataNum']
        t_time = mat_vals['tTime']
        return datas, data_num, t_time

    def plot_matrix(self, matrix):
        import matplotlib.pyplot as plt
        plt.plot(matrix)
        plt.show()

    def run(self):
        print "lfrm"
        datas, data_num, t_time = self.load_data('../data/enrondata.mat')
        # datas = self.load_data('../data/nips_1-17.mat')
        # data = numpy.array([[0,0,1,0,1,0,0,1,0],[0,0,0,0,0,0,1,0,1],[0,0,1,0,0,0,1,0,1],[0,1,1,0,0,0,0,1,0],
        #                    [0,0,0,0,0,0,1,0,1],[0,1,1,0,1,0,0,1,0],[1,0,0,0,0,1,0,0,0],[0,0,0,0,0,1,1,0,1],[1,0,0,1,0,1,0,0,0]])

        """
        data = numpy.array([[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                            [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                            [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                            # data = numpy.array([[1, 0, 1], [0, 0, 1], [1, 1, 0]])
        """
        self.initialize_data(datas[:, :, 0])
        #self.initialize_data(data)

        matrix_z = self.initialize_matrix_z()
        vec = self.sample(1000)
        print vec
        import matplotlib.pyplot as P
        from util.scaled_image import scaledimage

        # scaledimage(self.matrix_y)
        # scaledimage(self.matrix_z)
        # scaledimage(self.matrix_w)
        # scaledimage(self.log_likelihood_y(self.matrix_y, self.matrix_z, self.matrix_w))
        # P.show()

        # print (self.dim_n, self.dim_n)
        print self.matrix_z.shape
        title('No of items in Cluster')
        a = ""
        #for i in range(0, self.dim_k):
        #    a = a + vec[i]
        #    a =
        plot(vec[1])
        show()
        # print matrix_z

    def compute_probit(self, x):
        y = math.sqrt(2)*erfinv(2*x - 1)
        return y


if __name__ == '__main__':
    lfrm = LFRM()
    lfrm.run()
    #print scipy.stats.norm.cdf(-0.64499744)
    #print lfrm.compute_probit(0.025)
    #print rtnorm.rtnorm(0, 1, 0, 1, 5)
