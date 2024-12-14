import math
import numpy as np


class Transformation:
    """
    Represents spatial transformation parameters for aligning sections.
    """

    def __init__(self, tx=0.0, ty=0.0, scale=1.0, angle=0.0):
        self.tx = tx
        self.ty = ty
        self.scale = scale
        self.angle = angle

    def apply(self, coords):
        """
        Applies the transformation to a set of coordinates.
        """
        R = np.array([[math.cos(self.angle), -math.sin(self.angle)],
                      [math.sin(self.angle), math.cos(self.angle)]])
        return (self.scale * (R @ coords.T).T) + np.array([self.tx, self.ty])