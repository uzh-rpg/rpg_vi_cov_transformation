#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import transformations as tfs  # [w, x, y, z]
from tfs_utils import exp_so3, log_so3
import align_trajectory as align
import trace_parser as tp


# utility functions
def _assertDim(states):
    # 10: without bias; 16: with bias
    assert states.shape[1] == 10 or states.shape[1] == 16


def _getIndices(n_aligned, total_n):
    if n_aligned == -1:
        idxs = np.arange(0, total_n)
    else:
        assert n_aligned <= total_n and n_aligned >= 1
        idxs = np.arange(0, n_aligned)
    return idxs


# align by a 4-DoF transformation
def alignPositionYawSingleExpm(est, gt):
    '''
    calcualte the 4DOF transformation: yaw R and translation t so that:
        gt = R * est + t
    '''

    _assertDim(est)
    _assertDim(gt)
    est_pos, est_quat, est_vel = tp.parseSingle(est[0, :])
    g_pos, g_quat, g_vel = tp.parseSingle(gt[0, :])
    g_rot = tfs.quaternion_matrix(g_quat)
    g_rot = g_rot[0:3, 0:3]
    est_rot = tfs.quaternion_matrix(est_quat)
    est_rot = est_rot[0:3, 0:3]

    R_full = np.dot(g_rot, np.transpose(est_rot))
    log_R = log_so3(R_full)
    yaw_R = np.array([0, 0, log_R[2]])
    if np.linalg.norm(yaw_R) < 1e-7:
        R = np.identity(3)
    else:
        R = exp_so3(yaw_R)
    t = g_pos - np.dot(R, est_pos)

    return R, t


def alignPositionYaw(est, gt, n_aligned=1):
    _assertDim(est)
    _assertDim(gt)
    if n_aligned == 1:
        R, t = alignPositionYawSingleExpm(est, gt)
        return R, t
    else:
        idxs = _getIndices(n_aligned, est.shape[0])
        est_pos = est[idxs, 0:3]
        gt_pos = gt[idxs, 0:3]
        R, t = align.align_se3(gt_pos, est_pos,
                               only_yaw=True)  # note the order
        t = np.array(t)
        t = t.reshape((3, ))
        R = np.array(R)
        return R, t


# align by a SE3 transformation
def alignSE3Single(est, gt):
    '''
    Calculate SE3 transformation R and t so that:
        gt = R * est + t
    Using only the first poses of est and gt
    '''
    _assertDim(est)
    _assertDim(gt)
    est_pos, est_quat, est_vel = tp.parseSingle(est[0, :])
    g_pos, g_quat, g_vel = tp.parseSingle(gt[0, :])

    g_rot = tfs.quaternion_matrix(g_quat)
    g_rot = g_rot[0:3, 0:3]
    est_rot = tfs.quaternion_matrix(est_quat)
    est_rot = est_rot[0:3, 0:3]

    R = np.dot(g_rot, np.transpose(est_rot))
    t = g_pos - np.dot(R, est_pos)

    return R, t


def alignSE3(est, gt, n_aligned=-1):
    '''
    Calculate SE3 transformation R and t so that:
        gt = R * est + t
    '''
    _assertDim(est)
    _assertDim(gt)
    if n_aligned == 1:
        R, t = alignSE3Single(est, gt)
        return R, t
    else:
        idxs = _getIndices(n_aligned, est.shape[0])
        est_pos = est[idxs, 0:3]
        gt_pos = gt[idxs, 0:3]
        R, t = align.align_se3(gt_pos, est_pos)  # note the order
        t = np.array(t)
        t = t.reshape((3, ))
        R = np.array(R)
        return R, t


# transformations
def transformStates(states, R, t, scale=1):
    states_new = np.zeros(states.shape)
    for i, e in enumerate(states):
        p, q, v = tp.parseSingle(e)
        p_new = scale * np.dot(R, p) + t
        v_new = np.dot(R, v)
        rot = tfs.quaternion_matrix(q)
        rot = rot[0:3, 0:3]
        q_new = tfs.quaternion_from_matrix(np.dot(R, rot))
        states_new[i, 0:10] = np.hstack((p_new, q_new, v_new))

    return states_new


def transformPoints(pts, R, t):
    pts_new = np.zeros(pts.shape)
    for i, pt in enumerate(pts):
        pt_new = np.dot(R, pt) + t
        pts_new[i] = pt_new

    return pts_new


def transformRinc(incs, R):
    '''
    Exp(inc_new) = R Exp(inc)
    '''
    assert incs.shape[1] == 3

    new_incs = np.zeros(incs.shape)
    for i, inc in enumerate(incs):
        R_inc = exp_so3(inc)
        R_new_inc = np.dot(R, R_inc)
        new_inc = log_so3(R_new_inc)
        new_incs[i, :] = new_inc

    return new_incs


if __name__ == '__main__':
    pass
