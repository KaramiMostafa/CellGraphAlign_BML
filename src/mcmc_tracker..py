import numpy as np
from scipy.stats import norm, multivariate_normal
from .utils import propose_M, propose_theta


class MCMCTracker:
    """
    Implements a probabilistic MCMC-based cell tracking system.
    """

    def __init__(self, sections):
        self.sections = sections
        self.M_list = self._initialize_M()
        self.theta_list = self._initialize_theta(len(sections))

    def _initialize_M(self):
        """
        Initializes matching matrices based on nearest centroid.
        """
        M_list = []
        for t in range(len(self.sections) - 1):
            coords_t = self.sections[t].coords
            coords_t1 = self.sections[t + 1].coords
            tree_t1 = cKDTree(coords_t1)
            M = np.zeros((coords_t.shape[0], coords_t1.shape[0]))
            for i in range(coords_t.shape[0]):
                _, idx = tree_t1.query(coords_t[i])
                M[i, idx] = 1
            M_list.append(M)
        return M_list

    def _initialize_theta(self, T):
        """
        Initializes transformation parameters.
        """
        return [Transformation() for _ in range(T - 1)]

    def run(self, max_iter=3):
        """
        Runs the MCMC-based cell tracking.
        """
        for _ in range(max_iter):
            self._sample_M()
            self._sample_theta()
            print(f"Log-Posterior: {self._log_posterior()}")

    def _sample_M(self):
        """
        Samples matching matrices using MCMC.
        """
        for t in range(len(self.sections) - 1):
            M_current = self.M_list[t]
            M_proposed = propose_M(M_current)
            if self._acceptance_ratio_M(M_current, M_proposed, t) > np.random.rand():
                self.M_list[t] = M_proposed

    def _sample_theta(self):
        """
        Samples transformation parameters using MCMC.
        """
        for t in range(len(self.sections) - 1):
            theta_current = self.theta_list[t]
            theta_proposed = propose_theta(theta_current)
            if self._acceptance_ratio_theta(theta_current, theta_proposed, t) > np.random.rand():
                self.theta_list[t] = theta_proposed

    def _log_pair_posterior(self, M, theta, t):
        """
        Computes the posterior probability for a pair of sections.
        """
        likelihood_sum = sum(
            np.log(self._likelihood_of_match(i, j, self.sections[t], self.sections[t + 1], theta))
            for i in range(M.shape[0])
            for j in range(M.shape[1]) if M[i, j] == 1
        )
        return likelihood_sum