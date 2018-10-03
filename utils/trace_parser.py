#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import os


def parseTrajRes(filename):
    trace = open(filename, 'r')
    num_trials = int(trace.readline())

    mc_est = []
    mc_gt = []
    mc_pts_est = []
    mc_pts_gt = []
    mc_cov = []
    mc_iter_num = []
    mc_reproj_err = []
    mc_rot_err = []
    mc_vel_err = []
    mc_pos_err = []
    mc_prior_yaw_err = []  # position error
    mc_prior_pos_err = []  # rotation error
    mc_time_sec = []
    mc_q_inc = []
    for i in range(num_trials):
        num_kfs = int(trace.readline())
        est_single = np.zeros((num_kfs, 10))  # 3 + 4 + 3
        gt_single = np.zeros((num_kfs, 10))
        for j in range(num_kfs):
            line = (trace.readline()).split()
            est_single[j, :] = [float(e) for e in line]
        for j in range(num_kfs):
            line = (trace.readline()).split()
            gt_single[j, :] = [float(e) for e in line]

        num_pts_est = int(trace.readline())
        pts_est_single = np.zeros((num_pts_est, 3))
        for j in range(num_pts_est):
            line = (trace.readline()).split()
            pts_est_single[j, :] = [float(e) for e in line]

        num_pts_gt = int(trace.readline())
        assert num_pts_gt == num_pts_est
        pts_gt_single = np.zeros((num_pts_gt, 3))
        for j in range(num_pts_gt):
            line = (trace.readline()).split()
            pts_gt_single[j, :] = [float(e) for e in line]

        cov_rows = int(trace.readline())
        cov_cols = int(trace.readline())
        cov_single = np.zeros((cov_rows, cov_cols))
        for j in range(cov_rows):
            line = (trace.readline()).split()
            cov_single[j, :] = [float(e) for e in line]

        iter_num = int(trace.readline())
        line = (trace.readline()).split()
        reproj_err = [float(e) for e in line]

        trace.readline()
        line = (trace.readline()).split()
        preinte_err = [float(e) for e in line]

        trace.readline()
        line = (trace.readline()).split()
        prior_err = [float(e) for e in line]

        time_sec = float(trace.readline())

        trace.readline()
        line = (trace.readline()).split()
        q_inc = np.array([float(e) for e in line])
        q_inc = q_inc.reshape((num_kfs, 3))

        mc_est.append(est_single)
        mc_gt.append(gt_single)
        mc_pts_est.append(pts_est_single)
        mc_pts_gt.append(pts_gt_single)
        mc_cov.append(cov_single)
        mc_iter_num.append(iter_num)
        mc_reproj_err.append(reproj_err)
        mc_rot_err.append(preinte_err[0::3])
        mc_vel_err.append(preinte_err[1::3])
        mc_pos_err.append(preinte_err[2::3])
        mc_prior_yaw_err.append(prior_err[0::2])
        mc_prior_pos_err.append(prior_err[1::2])

        mc_time_sec.append(time_sec)

        mc_q_inc.append(q_inc)

    trace.close()
    mc_res = {'mc_est': mc_est, 'mc_gt': mc_gt,
              'mc_pts_est': mc_pts_est, 'mc_pts_gt': mc_pts_gt,
              'mc_cov': mc_cov,
              'mc_iter': mc_iter_num, 'mc_reproj_err': mc_reproj_err,
              'mc_rot_err': mc_rot_err, 'mc_vel_err': mc_vel_err,
              'mc_pos_err': mc_pos_err,
              'mc_prior_yaw_err': mc_prior_yaw_err,
              'mc_prior_pos_err': mc_prior_pos_err,
              'mc_time_sec': mc_time_sec,
              'mc_q_inc': mc_q_inc}
    return mc_res


def parseTrajResBatch(trace_dir):
    catalog = open(os.path.join(trace_dir, 'trajectories.txt'), 'r')
    traj_files = [os.path.join(trace_dir, l[0: -1]) for l in catalog]
    mc_res_vec = []
    for i, traj in enumerate(traj_files):
        mc_res = parseTrajRes(traj)
        mc_res_vec.append(mc_res)
    return mc_res_vec


def parseSingle(l):
    return l[0:3], l[3:7], l[7:10]
