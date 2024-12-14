import numpy as np
from scipy.spatial import cKDTree


class CellGraph:
    """
    Represents a graph of cells within a single microscopy section.
    """

    def __init__(self, coords, areas, perimeters, distances, names):
        self.coords = coords
        self.areas = areas
        self.perimeters = perimeters
        self.distances = distances
        self.names = names
        self.graph = self._build_graph(coords)

    def _build_graph(self, coords, k=5):
        """
        Builds a graph based on k-nearest neighbors.
        """
        n = coords.shape[0]
        if n < 2:
            return [set() for _ in range(n)]
        k = min(k, n - 1)
        tree = cKDTree(coords)
        _, idxs = tree.query(coords, k=k + 1)
        return [set(i_row[1:]) for i_row in idxs]