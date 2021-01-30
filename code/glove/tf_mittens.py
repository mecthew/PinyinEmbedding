"""tf_mittens.py

Fast implementations of Mittens and GloVe in Tensorflow.

See https://nlp.stanford.edu/pubs/glove.pdf for details of GloVe.

References
----------
[1] Jeffrey Pennington, Richard Socher, and Christopher D. Manning.
2014. GloVe: Global Vectors for Word Representation

[2] Nick Dingwall and Christopher Potts. 2018. Mittens: An Extension
of GloVe for Learning Domain-Specialized Representations

Authors: Nick Dingwall, Chris Potts
"""
import os

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from mittens_base import randmatrix, noise
from mittens_base import MittensBase, GloVeBase


_FRAMEWORK = "TensorFlow"
_DESC = """
    This version is faster than the NumPy version. If you prefer NumPy
    for some reason, import it directly: 
    
    >>> from mittens.np_mittens import {model}
    """


class Mittens(MittensBase):

    __doc__ = MittensBase.__doc__.format(
        framework=_FRAMEWORK,
        second=_DESC.format(model=MittensBase._MODEL))

    @property
    def framework(self):
        return _FRAMEWORK

    def _fit(self, X, weights, log_coincidence,
             vocab=None,
             initial_embedding_dict=None,
             fixed_initialization=None):
        if fixed_initialization is not None:
            raise AttributeError("Tensorflow version of Mittens does "
                                 "not support specifying initializations.")

        # Start the session:
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()

        # Build the computation graph.
        self._build_graph(vocab, initial_embedding_dict)

        # Optimizer set-up:
        self.cost = self._get_cost_function()
        self.optimizer = self._get_optimizer()

        # Set up logging for Tensorboard
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            n_subdirs = len(os.listdir(self.log_dir))
            subdir = self.log_subdir or str(n_subdirs + 1)
            directory = os.path.join(self.log_dir, subdir)
            log_writer = tf.summary.FileWriter(directory, flush_secs=1)

        # Run training
        self.sess.run(tf.global_variables_initializer())
        if self.test_mode:
            self.W_start = self.sess.run(self.W)
            self.C_start = self.sess.run(self.C)
            self.bw_start = self.sess.run(self.bw)
            self.bc_start = self.sess.run(self.bc)

        merged_logs = tf.summary.merge_all()
        train_iter = tqdm(range(self.max_iter + 1))
        for i in train_iter:  # 保存字最后一轮数据到summary
            _, loss, stats = self.sess.run(
                [self.optimizer, self.cost, merged_logs],
                feed_dict={
                    self.weights: weights,
                    self.log_coincidence: log_coincidence})

            # Keep track of losses
            if self.log_dir and i % 10 == 0:
                log_writer.add_summary(stats, global_step=i)
            self.errors.append(loss)

            if loss < self.tol:
                # Quit early if tolerance is met
                train_iter.set_description("Iteration {}: stopping with loss < self.tol".format(i))
                # self._progressbar("stopping with loss < self.tol", i - 1)
                break
            else:
                train_iter.set_description("Iteration {}: loss: {}".format(i, loss))
                # self._progressbar("loss: {}".format(loss), i - 1)

        # 绘制训练过程中的loss变化，tensorboard也有记载
        plt.figure()
        plt.plot(np.arange(len(self.errors)), self.errors)
        plt.show()

        # Return the sum of the two learned matrices, as recommended
        # in the paper:
        return self.sess.run(tf.add(self.W, self.C))

    def _build_graph(self, vocab, initial_embedding_dict):
        """Builds the computatation graph.

        Parameters
        ------------
        vocab : Iterable
        initial_embedding_dict : dict
        """
        # Constants
        self.ones = tf.ones([self.n_words, 1])

        # Parameters:
        if initial_embedding_dict is None:
            # Ordinary GloVe
            self.W = self._weight_init(self.n_words, self.n, 'W')
            self.C = self._weight_init(self.n_words, self.n, 'C')
        else:
            # This is the case where we have values to use as a
            # "warm start":
            assert self.n == len(next(iter(initial_embedding_dict.values())))
            W = randmatrix(len(vocab), self.n)
            C = randmatrix(len(vocab), self.n)
            self.original_embedding = np.zeros((len(vocab), self.n))
            self.has_embedding = np.zeros(len(vocab), dtype=np.bool)
            for i, w in enumerate(vocab):
                if w in initial_embedding_dict:
                    self.has_embedding[i] = 1
                    embedding = np.array(initial_embedding_dict[w])
                    self.original_embedding[i] = embedding
                    # Divide the original embedding into W and C,
                    # plus some noise to break the symmetry that would
                    # otherwise cause both gradient updates to be
                    # identical.
                    W[i] = 0.5 * embedding + noise(self.n)
                    C[i] = 0.5 * embedding + noise(self.n)
            self.W = tf.Variable(W, name='W', dtype=tf.float32)
            self.C = tf.Variable(C, name='C', dtype=tf.float32)
            self.original_embedding = tf.constant(self.original_embedding,
                                                  dtype=tf.float32)
            self.has_embedding = tf.constant(self.has_embedding,
                                             dtype=tf.bool)
            # This is for testing. It differs from
            # `self.original_embedding` only in that it includes the
            # random noise we added above to break the symmetry.
            self.G_start = W + C

        self.bw = self._weight_init(self.n_words, 1, 'bw')
        self.bc = self._weight_init(self.n_words, 1, 'bc')

        self.model = tf.matmul(self.W, tf.transpose(self.C)) + self.bw + tf.transpose(self.bc)
        # self.model = tf.tensordot(self.W, tf.transpose(self.C), axes=1) + \
        #              tf.tensordot(self.bw, tf.transpose(self.ones), axes=1) + \
        #              tf.tensordot(self.ones, tf.transpose(self.bc), axes=1)

    def _get_cost_function(self):
        """Compute the cost of the Mittens objective function.

        If self.mittens = 0, this is the same as the cost of GloVe.
        """
        self.weights = tf.placeholder(
            tf.float32, shape=[self.n_words, self.n_words])
        self.log_coincidence = tf.placeholder(
            tf.float32, shape=[self.n_words, self.n_words])
        self.diffs = tf.subtract(self.model, self.log_coincidence)
        cost = tf.reduce_sum(
            0.5 * tf.multiply(self.weights, tf.square(self.diffs)))
        if self.mittens > 0:
            self.mittens = tf.constant(self.mittens, tf.float32)
            cost += self.mittens * tf.reduce_sum(
                        tf.square(tf.subtract(tf.add(self.W, self.C)[self.has_embedding],
                        self.original_embedding[self.has_embedding])))
        tf.summary.scalar("cost", cost)
        return cost

    @staticmethod
    def _tf_squared_euclidean(X, Y):
        """Squared Euclidean distance between the rows of `X` and `Y`.
        """
        return tf.reduce_sum(tf.square(tf.subtract(X, Y)), axis=1)

    def _get_optimizer(self):
        """Uses Adagrad to optimize the GloVe/Mittens objective,
        as specified in the GloVe paper.
        """
        optim = tf.train.AdagradOptimizer(self.learning_rate)
        gradients = optim.compute_gradients(self.cost)
        if self.log_dir:
            for name, (g, v) in zip(['W', 'C', 'bw', 'bc'], gradients):
                tf.summary.histogram("{}_grad".format(name), g)
                tf.summary.histogram("{}_vals".format(name), v)
        return optim.apply_gradients(gradients)

    def _weight_init(self, m, n, name):
        """
        Uses the Xavier Glorot method for initializing weights. This is
        built in to TensorFlow as `tf.contrib.layers.xavier_initializer`,
        but it's nice to see all the details.
        """
        x = np.sqrt(6.0/(m+n))
        return tf.Variable(
            tf.random_uniform(
                [m, n], minval=-x, maxval=x), name=name)


class GloVe(Mittens, GloVeBase):

    __doc__ = GloVeBase.__doc__.format(
        framework=_FRAMEWORK,
        second=_DESC.format(model=GloVeBase._MODEL))
