#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np

import utils.trace_parser as tp
import utils.tfs_utils as tu
import utils.align_utils as au


def transform_vi_covariance(cov_free,
                            q_inc_free, q_inc_fixed,
                            est_free, est_fixed,
                            all_inc_rot=False):
    """Transform the free gauge covariance to the same reference frame as
       the gauge fixation one

    Suppose N is the number of keyframes.

    Input:
    cov_free -- 9N x 9N array
                The free gauge covariance to be transformed.
                The states are ordered as [position, rotation, velocity]

    q_inc_free -- list of length N, each of which is a (3,) numpy array.
                  The estimated incremental rotation of the free gauge method.
                  If all_inc_rot is False, only the first element is used.

    q_inc_fixed -- list of length N, each of which is a (3,) numpy array
                   The estimated incremental rotations of the fixed gauge
                   method. If all_inc_rot is False, only the first element is
                   meaningful and used.

    est_free -- list of length N, each of which is a (10,) numpy array.
                The estimated state of the free gauge method.
                Each state is ordered as [position, rotation, velocity]

    est_fixed -- list of length N, each of which is a (10,) numpy array.
                 The estimated state of the fixed gauge method.
                 Each state is ordered as [position, rotation, velocity]

    all_inc_rot -- If this parameter is True, the incremental rotation
                   parameterization is used for all rotations.
                   Otherwise, it is only used for the first rotation.
                   This affects the transformation of the rotation-related
                   covariance.

    Output:
    cov_proj -- The final transformed covaraince (9N x 9N)
    cov_aligend -- The intermediate covariance before the oblique projection.
    Q -- The oblique projection matrix.

    """
    n_states = cov_free.shape[0]
    assert n_states % 9 == 0
    n_kfs = n_states / 9

    pos_from_0, quat_from_0, vel_from_0 = tp.parseSingle(est_free[0, :])
    pos_to_0, quat_to_0, vel_to_0 = tp.parseSingle(est_fixed[0, :])

    print("Going to transform covariance of {0} keyframes".format(n_kfs))

    # 0 calcualte transformation from the first state
    R_inc_0_from = tu.exp_so3(q_inc_free[0])
    R_inc_0_to = tu.exp_so3(q_inc_fixed[0])

    R_full = np.dot(R_inc_0_to, np.transpose(R_inc_0_from))
    log_R = tu.log_so3(R_full)
    yaw_R = np.array([0, 0, log_R[2]])
    if np.linalg.norm(yaw_R) < 1e-7:
        R = np.identity(3)
    else:
        R = tu.exp_so3(yaw_R)
    t = pos_to_0 - np.dot(R, pos_from_0)

    print("The transformation to be applied is\n{0}\n{1}".format(R, t))

    # align the estimation
    q_inc_aligned = au.transformRinc(q_inc_free, R)
    est_aligned = au.transformStates(est_free, R, t)

    # 1 transform the covariance
    tfs_jac = np.zeros((9*n_kfs, 9*n_kfs))
    for j in range(n_kfs):
        cs = j * 3

        # positions
        tfs_jac[cs:cs+3, cs:cs+3] = R

        # velocity
        tfs_jac[6*n_kfs + cs: 6*n_kfs + cs+3,
                6*n_kfs + cs: 6*n_kfs + cs+3] = R

        # q_inc
        if all_inc_rot or j == 0:
            inv_Jr = tu.invRightJacobian(q_inc_aligned[j])
            Jr = tu.rightJacobian(q_inc_free[j])
            tfs_jac[3*n_kfs + cs: 3*n_kfs + cs+3,
                    3*n_kfs + cs: 3*n_kfs + cs+3] = np.dot(inv_Jr, Jr)
        else:
            tfs_jac[3*n_kfs + cs: 3*n_kfs + cs+3,
                    3*n_kfs + cs: 3*n_kfs + cs+3] = R

    cov_aligned = np.dot(tfs_jac,
                         np.dot(cov_free, np.transpose(tfs_jac)))

    # 2 project the covariance
    az = np.array([0, 0, 1])
    s_az = tu.skewv3(az)
    # U
    U = np.zeros((n_states, 4))
    for j in range(n_kfs):
        cs = 3 * j
        # position
        U[cs: cs + 3, 0] = np.dot(s_az, est_aligned[j, 0:3])
        U[cs: cs + 3, 1:4] = np.identity(3)

        # rotation
        if j == 0 or all_inc_rot:
            U[3*n_kfs+cs: 3*n_kfs+cs+3, 0] = \
                np.dot(tu.invLeftJacobian(q_inc_aligned[j]), az)
            U[3*n_kfs+cs: 3*n_kfs+cs + 3, 1:4] = np.zeros((3, 3))
        else:
            U[3*n_kfs+cs: 3*n_kfs+cs+3, 0] = az
            U[3*n_kfs+cs: 3*n_kfs+cs + 3, 1:4] = np.zeros((3, 3))

        # velocity
        U[6*n_kfs+cs: 6*n_kfs+cs+3, 0] = \
            np.dot(s_az, est_aligned[j, 7:10])

        U[6*n_kfs+cs: 6*n_kfs+cs + 3, 1:4] = np.zeros((3, 3))

    # V
    V = np.zeros((n_states, 4))
    V[0:3, 1:4] = np.identity(3)
    V[3*n_kfs + 2, 0] = 1

    VTU = np.dot(np.transpose(V), U)
    invVTU = np.linalg.inv(VTU)
    print("VTU is\n{0} \nand its inverse is\n{1}".format(VTU, invVTU))
    Q = np.identity(n_states) - np.dot(U, np.dot(invVTU, np.transpose(V)))

    print("The rank of Q is {0} of size {1}".format(np.linalg.matrix_rank(Q),
                                                    Q.shape[0]))

    cov_proj = np.dot(Q, np.dot(cov_aligned, np.transpose(Q)))

    return cov_proj, cov_aligned, Q

if __name__ == '__main__':
    pass
