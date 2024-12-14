import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm, multivariate_normal
import math

class CellGraph:
    """
    Represents a graph of cells within a single microscopy section.
    """

    def __init__(self, coords, areas, perimeters, distances, names):
        """
        Initializes a CellGraph instance.

        :param coords: Array of centroid coordinates for cells.
        :param areas: Array of cell areas.
        :param perimeters: Array of cell perimeters.
        :param distances: Array of distances to the nearest neighbor.
        :param names: Array of cell names.
        """
        self.coords = coords
        self.areas = areas
        self.perimeters = perimeters
        self.distances = distances
        self.names = names
        self.graph = self._build_graph(coords)

    def _build_graph(self, coords, k=5):
        n = coords.shape[0]
        if n < 2:
            return [set() for _ in range(n)]
        k = min(k, n - 1)
        tree = cKDTree(coords)
        _, idxs = tree.query(coords, k=k + 1)
        return [set(i_row[1:]) for i_row in idxs]


class Transformation:
    """
    Represents spatial transformation parameters for aligning sections.
    """

    def __init__(self, tx=0.0, ty=0.0, scale=1.0, angle=0.0):
        """
        Initializes a Transformation instance.

        :param tx: Translation along the x-axis.
        :param ty: Translation along the y-axis.
        :param scale: Scaling factor.
        :param angle: Rotation angle in radians.
        """
        self.tx = tx
        self.ty = ty
        self.scale = scale
        self.angle = angle

    def apply(self, coords):
        """
        Applies the transformation to a set of coordinates.

        :param coords: Input coordinates as a 2D array.
        :return: Transformed coordinates.
        """
        R = np.array([[math.cos(self.angle), -math.sin(self.angle)],
                      [math.sin(self.angle), math.cos(self.angle)]])
        return (self.scale * (R @ coords.T).T) + np.array([self.tx, self.ty])


class MCMCTracker:
    """
    Implements a probabilistic MCMC-based cell tracking system.
    """

    def __init__(self, sections):
        """
        Initializes the MCMCTracker with a set of sections.

        :param sections: List of CellGraph instances representing consecutive sections.
        """
        self.sections = sections
        self.M_list = self._initialize_M()
        self.theta_list = self._initialize_theta(len(sections))

    def _initialize_M(self):
        """
        Initializes the matching matrix for all sections based on nearest centroid.

        :return: List of initial matching matrices.
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
        Initializes transformation parameters for all section pairs.

        :param T: Number of sections.
        :return: List of Transformation instances.
        """
        return [Transformation() for _ in range(T - 1)]

    def _likelihood_of_match(self, i, j, section_t, section_t1, theta):
        c_i = section_t.coords[i]
        c_j = section_t1.coords[j]
        A_i, P_i, D_i = section_t.areas[i], section_t.perimeters[i], section_t.distances[i]
        A_j, P_j, D_j = section_t1.areas[j], section_t1.perimeters[j], section_t1.distances[j]

        c_i_trans = theta.apply(np.atleast_2d(c_i))[0]
        pos_diff = c_j - c_i_trans
        pos_prob = multivariate_normal.pdf(pos_diff, mean=[0, 0], cov=[[50, 0], [0, 50]])
        A_prob = norm.pdf(A_j, loc=A_i, scale=10)
        P_prob = norm.pdf(P_j, loc=P_i, scale=10)
        D_prob = norm.pdf(D_j, loc=D_i, scale=10)

        return max(pos_prob * A_prob * P_prob * D_prob, 1e-20)

    def run(self, max_iter=3):
        """
        Executes the MCMC-based cell tracking process.

        :param max_iter: Number of iterations for the MCMC process.
        """
        for _ in range(max_iter):
            self._sample_M()
            self._sample_theta()
            print(f"Log-Posterior: {self._log_posterior()}")

    def _sample_M(self):
        """
        Samples matching matrices (M) for all section pairs using MCMC.
        """
        for t in range(len(self.sections) - 1):
            M_current = self.M_list[t]
            M_proposed = self._propose_M(M_current)
            if self._acceptance_ratio_M(M_current, M_proposed, t) > np.random.rand():
                self.M_list[t] = M_proposed

    def _sample_theta(self):
        """
        Samples transformation parameters (theta) for all section pairs using MCMC.
        """
        for t in range(len(self.sections) - 1):
            theta_current = self.theta_list[t]
            theta_proposed = self._propose_theta(theta_current)
            if self._acceptance_ratio_theta(theta_current, theta_proposed, t) > np.random.rand():
                self.theta_list[t] = theta_proposed

    def _propose_M(self, M):
        M_new = M.copy()
        i = np.random.randint(M.shape[0])
        j = np.random.randint(M.shape[1])
        M_new[i, :] = 0
        M_new[i, j] = 1
        return M_new

    def _propose_theta(self, theta):
        return Transformation(
            tx=theta.tx + np.random.normal(0, 5),
            ty=theta.ty + np.random.normal(0, 5),
            scale=max(0.5, min(2.0, theta.scale + np.random.normal(0, 0.1))),
            angle=theta.angle + np.random.normal(0, 0.1)
        )

    def _acceptance_ratio_M(self, M_current, M_proposed, t):
        lp_current = self._log_pair_posterior(M_current, self.theta_list[t], t)
        lp_proposed = self._log_pair_posterior(M_proposed, self.theta_list[t], t)
        return min(1, np.exp(lp_proposed - lp_current))

    def _acceptance_ratio_theta(self, theta_current, theta_proposed, t):
        lp_current = self._log_pair_posterior(self.M_list[t], theta_current, t)
        lp_proposed = self._log_pair_posterior(self.M_list[t], theta_proposed, t)
        return min(1, np.exp(lp_proposed - lp_current))

    def _log_pair_posterior(self, M, theta, t):
        likelihood_sum = sum(
            np.log(self._likelihood_of_match(i, j, self.sections[t], self.sections[t + 1], theta))
            for i in range(M.shape[0])
            for j in range(M.shape[1]) if M[i, j] == 1
        )
        return likelihood_sum


if __name__ == "__main__":
    # Example usage
    data = pd.read_csv("/CellGraphAlign_BML/multisectionmeasurements.csv")
    sections = [
        CellGraph(
            coords=section_data[['Centroid X µm', 'Centroid Y µm']].values,
            areas=section_data['Area'].values,
            perimeters=section_data['Perimeter'].values,
            distances=section_data['Distance in um to nearest Cell'].values,
            names=section_data['Name'].values
        )
        for _, section_data in data.groupby('Image')
    ]

    tracker = MCMCTracker(sections)
    tracker.run(max_iter=3)