import numpy as np


def propose_M(M):
    """
    Proposes a new matching matrix by making a local change.
    """
    M_new = M.copy()
    i = np.random.randint(M.shape[0])
    j = np.random.randint(M.shape[1])
    M_new[i, :] = 0
    M_new[i, j] = 1
    return M_new


def propose_theta(theta):
    """
    Proposes a new transformation by adding Gaussian noise.
    """
    return Transformation(
        tx=theta.tx + np.random.normal(0, 5),
        ty=theta.ty + np.random.normal(0, 5),
        scale=max(0.5, min(2.0, theta.scale + np.random.normal(0, 0.1))),
        angle=theta.angle + np.random.normal(0, 0.1)
    )