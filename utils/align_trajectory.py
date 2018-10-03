#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from tfs_utils import exp_so3, log_so3


def align_se3(model, data, only_yaw=False):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (nx3), numpy array type
    data -- second trajectory (nx3), numpy array type
    only_yaw -- constrain the rotation to yaw only

    Output:
    R -- rotation matrix (3x3)
    t -- translation vector (3x1)

    """
    np.set_printoptions(precision=3, suppress=True)
    mu_M = model.mean(0)
    mu_D = data.mean(0)
    model_zerocentered = model - mu_M
    data_zerocentered = data - mu_D

    W = np.zeros((3, 3))
    for row in range(model.shape[0]):
        W += np.outer(model_zerocentered[row, :], data_zerocentered[row, :])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    R = U*S*Vh
    if only_yaw:
        log_R = log_so3(R)
        yaw_R = np.array([0, 0, log_R[2]])
        R = exp_so3(yaw_R)
    t = mu_D - np.dot(R, mu_M)

    return R, t
