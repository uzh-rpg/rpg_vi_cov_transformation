#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np


def permuteHesCov(cov, num_kf):
    '''
    permutate the covriance/hessian matrix so that
    positions, rotations and velocities appear together.
    assum that the states are ordered [p0, q0, v0; p1, q1, v1; ...]
    '''
    assert cov.shape[0] == 9 * num_kf
    num_states = 9 * num_kf

    des_index = np.zeros((num_states, ))
    for i in range(num_kf):
        cur_pos_s = i * 3
        old_pos_s = i * 9
        des_index[cur_pos_s] = old_pos_s
        des_index[cur_pos_s+1] = old_pos_s+1
        des_index[cur_pos_s+2] = old_pos_s+2
        cur_rot_s = 3 * num_kf + i * 3
        old_rot_s = i * 9 + 3
        des_index[cur_rot_s] = old_rot_s
        des_index[cur_rot_s+1] = old_rot_s+1
        des_index[cur_rot_s+2] = old_rot_s+2
        cur_vel_s = 6 * num_kf + i * 3
        old_vel_s = i * 9 + 6
        des_index[cur_vel_s] = old_vel_s
        des_index[cur_vel_s+1] = old_vel_s+1
        des_index[cur_vel_s+2] = old_vel_s+2

    new_cov = cov
    cur_index = np.array(range(num_states))

    for i in range(num_states):
        des_id = des_index[i]
        loc_des_state = np.argwhere(cur_index == des_id)[0][0]
        if loc_des_state == i:
            continue
        # swap i and loc_des_state
        tmp = cur_index[i]
        cur_index[i] = cur_index[loc_des_state]
        cur_index[loc_des_state] = tmp
        new_cov[:, [i, loc_des_state]] = new_cov[:, [loc_des_state, i]]
        new_cov[[i, loc_des_state], :] = new_cov[[loc_des_state, i], :]

    return new_cov
