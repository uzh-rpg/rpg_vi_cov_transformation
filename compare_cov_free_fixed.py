#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import Ellipse

import utils.mat_utils as mu
import utils.trace_parser as tp

import cov_transformation as ct

rc('font', **{'family': 'serif', 'serif': ['Cardo'], 'size': 20})
rc('text', usetex=True)

# directories
cur_file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_file_dir, 'data/compare_cov_free_fix')
plot_dir = os.path.join(cur_file_dir, 'plots')


# NOTE: choose different datasets here
# results that used incremental rotation for ALL orientations
# all_inc_rot = True
# free_trace = os.path.join(data_dir, 'inc_rot_all/free_sine.txt')
# fix_trace = os.path.join(data_dir, 'inc_rot_all/fix_sine.txt')

# results that used incremental rotation ONLY for the first orientation
all_inc_rot = False
free_trace = os.path.join(data_dir, 'inc_rot_first/free_mh_30.txt')
fix_trace = os.path.join(data_dir, 'inc_rot_first/fix_mh_30.txt')

# parsing
print("#####################")
print("Parsing results...")
print("#####################")
mc_res_free = tp.parseTrajRes(free_trace)
mc_res_fix = tp.parseTrajRes(fix_trace)
num_trials = len(mc_res_free['mc_est'])
num_kfs = mc_res_free['mc_est'][0].shape[0]
num_sts = 9 * num_kfs

print("Current trace has {0} trials with {1} keyframes.".format(num_trials,
                                                                num_kfs))
valid_cov_ids = [0]
# permute covaraince: put position/rotation/velocity together
mc_perm_cov_free = []
for i in valid_cov_ids:
    cur_num_kfs = mc_res_free['mc_est'][i].shape[0]
    perm_cov = mu.permuteHesCov(
        mc_res_free['mc_cov'][i][0:9*num_kfs, 0:9*num_kfs], num_kfs)
    mc_perm_cov_free.append(perm_cov)
mc_res_free['mc_cov'] = mc_perm_cov_free

mc_perm_cov_fix = []
for i in valid_cov_ids:
    cur_num_kfs = mc_res_fix['mc_est'][i].shape[0]
    perm_cov = mu.permuteHesCov(
        mc_res_fix['mc_cov'][i][0:9*num_kfs, 0:9*num_kfs], num_kfs)
    mc_perm_cov_fix.append(perm_cov)
mc_res_fix['mc_cov'] = mc_perm_cov_fix

# visualization one trial
viz_id = 0
cov_free = mc_res_free['mc_cov'][viz_id]
cov_fix = mc_res_fix['mc_cov'][viz_id]
q_inc_free = mc_res_free['mc_q_inc'][viz_id]
q_inc_fix = mc_res_fix['mc_q_inc'][viz_id]
est_free = mc_res_free['mc_est'][viz_id]
est_fix = mc_res_fix['mc_est'][viz_id]

# transform covariance
print("#####################")
print("Transform covaraince for trial {0}...".format(viz_id))
print("#####################")
cov_free_transformed, cov_free_aligned, Q = \
    ct.transform_vi_covariance(cov_free,
                               q_inc_free, q_inc_fix,
                               est_free, est_fix, all_inc_rot)
gt_fixed = mc_res_fix['mc_gt'][viz_id]

# visualize the covaraince as matrices
print("#####################")
print("Plotting covariance matrices and saving results...")
print("#####################")
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(Q, interpolation='none', cmap='gray_r')
plt.colorbar()
fig.savefig(plot_dir + '/Q.pdf', bbox_inches='tight')

dim_sts = cov_free.shape[0]
num_kfs = dim_sts / 9
one_third = dim_sts / 3
print("Visualization: {0} states and {1} keyframes.".format(dim_sts, num_kfs))
ticks = [0.5 * one_third, 1.5 * one_third, 2.5 * one_third]
st_labels = ['$\mathbf{p}$', '$\phi$', '$\mathbf{v}$']

zero_region = np.zeros((dim_sts, dim_sts))
zero_region[:, 0:3] = 1.0
zero_region[0:3, :] = 1.0
zero_region[3*num_kfs+2, :] = 1.0
zero_region[:, 3*num_kfs+2] = 1.0
zero_region = np.ma.masked_where(zero_region < 0.5, zero_region)

disp_offset = 1e-6

fig = plt.figure()
fig.canvas.set_window_title('Free Covariance')
ax = fig.add_subplot(111)
plt.imshow(np.log10(disp_offset + np.abs(cov_free)), interpolation='none',
           cmap='gray_r')
plt.colorbar()
ax.set_xticks(ticks)
ax.set_xticklabels(st_labels)
ax.set_yticks(ticks)
ax.set_yticklabels(st_labels)
ax.tick_params(axis=u'both', which=u'both', length=0)
plt.tight_layout()
fig.savefig(plot_dir + '/free_cov_mat.pdf', bbox_inches='tight')

fig = plt.figure()
fig.canvas.set_window_title('Aligned Free Covariance')
ax = fig.add_subplot(111)
plt.imshow(np.log10(disp_offset + np.abs(cov_free_aligned)),
           interpolation='none', cmap='gray_r')
plt.colorbar()
ax.set_xticks(ticks)
ax.set_xticklabels(st_labels)
ax.set_yticks(ticks)
ax.set_yticklabels(st_labels)
ax.tick_params(axis=u'both', which=u'both', length=0)
plt.tight_layout()
fig.savefig(plot_dir + '/aligned_free_cov_mat.pdf', bbox_inches='tight')


cmap = 'autumn'

fig = plt.figure()
fig.canvas.set_window_title('Transformed Free Covariance')
ax = fig.add_subplot(111)
plt.imshow(np.log10(disp_offset + np.abs(cov_free_transformed)),
           interpolation='none', cmap='gray_r')
plt.colorbar()
plt.imshow(zero_region, interpolation='none', cmap=cmap, vmax=1.0, vmin=0.0, 
           alpha=1.0)
ax.set_xticks(ticks)
ax.set_xticklabels(st_labels)
ax.set_yticks(ticks)
ax.set_yticklabels(st_labels)
ax.tick_params(axis=u'both', which=u'both', length=0)
plt.tight_layout()
fig.savefig(plot_dir + '/transformed_cov_mat.pdf', bbox_inches='tight')

fig = plt.figure()
fig.canvas.set_window_title('Fixed Covariance')
ax = fig.add_subplot(111)
plt.imshow(np.log10(disp_offset + np.abs(cov_fix)), interpolation='none',
           cmap='gray_r')
plt.colorbar()
plt.imshow(zero_region, interpolation='none', cmap=cmap, vmax=1.0, vmin=0.0,
           alpha=1.0)
ax.set_xticks(ticks)
ax.set_xticklabels(st_labels)
ax.set_yticks(ticks)
ax.set_yticklabels(st_labels)
ax.tick_params(axis=u'both', which=u'both', length=0)
plt.tight_layout()
fig.savefig(plot_dir + '/fixed_cov_mat.pdf', bbox_inches='tight')

# visualize position uncertainties
print("#####################")
print("Plotting 2D position uncertainties and saving results...")
print("#####################")
c_free = 'r'
c_fixed = 'b'
free_viz_s = 1e3
free_align_var_ellips = []
for i in range(num_kfs):
    cur_var_e = Ellipse(gt_fixed[i, 0:2],
                        free_viz_s * cov_free_aligned[i*3, i*3],
                        free_viz_s * cov_free_aligned[i*3+1, i*3+1],
                        fill=False)
    free_align_var_ellips.append(cur_var_e)

fix_viz_s = 3e3
fix_var_ellips = []
for i in range(num_kfs):
    cur_var_e = Ellipse(gt_fixed[i, 0:2],
                        fix_viz_s * cov_fix[i*3, i*3],
                        fix_viz_s * cov_fix[i*3+1, i*3+1],
                        fill=False)
    fix_var_ellips.append(cur_var_e)

free_proj_var_ellips = []
for i in range(num_kfs):
    cur_var_e = Ellipse(gt_fixed[i, 0:2],
                        fix_viz_s * cov_free_transformed[i*3, i*3],
                        fix_viz_s * cov_free_transformed[i*3+1, i*3+1],
                        fill=False)
    free_proj_var_ellips.append(cur_var_e)

fig = plt.figure(figsize=(6, 6))
fig.canvas.set_window_title('Free 2D')
ax = fig.add_subplot(111)
ax.plot(gt_fixed[:, 0], gt_fixed[:, 1], color='k')
for e in free_align_var_ellips:
    ax.add_artist(e)
    e.set_facecolor('none')
    e.set_color(c_free)
    e.set(label='Free')
# ax.legend()
plt.tight_layout()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')
fig.savefig(plot_dir + '/aligned_free_cov.pdf', bbox_inches='tight')

fig = plt.figure(figsize=(6, 6))
fig.canvas.set_window_title('Fixed 2D')
ax = fig.add_subplot(111)
ax.plot(gt_fixed[:, 0], gt_fixed[:, 1], color='k')
for e in fix_var_ellips:
    ax.add_artist(e)
    e.set_facecolor('none')
    e.set_color(c_fixed)
    e.set_label('Fixed')
# ax.legend()
plt.tight_layout()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')
fig.savefig(plot_dir + '/fixed_cov.pdf', bbox_inches='tight')

fig = plt.figure(figsize=(6, 6))
fig.canvas.set_window_title('Transformed 2D')
ax = fig.add_subplot(111)
ax.plot(gt_fixed[:, 0], gt_fixed[:, 1], color='k')
for e in free_proj_var_ellips:
    ax.add_artist(e)
    e.set_facecolor('none')
    e.set_color(c_free)
    e.set_label('Transformed Free')
# ax.legend()
plt.tight_layout()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')
fig.savefig(plot_dir + '/transformed_free_cov.pdf', bbox_inches='tight')

# calculate statistics
print("#####################")
print("Calculating statistics...")
print("#####################")
transformed_diff_cov = cov_free_transformed - cov_fix
org_diff_cov = cov_free - cov_fix

n_org_diff_cov = np.linalg.norm(org_diff_cov, 'fro')
n_transformed_diff_cov = np.linalg.norm(transformed_diff_cov, 'fro')

n_org_free_cov = np.linalg.norm(cov_free, 'fro')
n_transformed_free_cov = np.linalg.norm(cov_free_transformed, 'fro')
n_fix_cov = np.linalg.norm(cov_fix, 'fro')

print("The Frobenius norm of covariance difference"
      " before transformation is {0}.".format(n_org_diff_cov))
print("The Frobenius norm of covariance difference"
      " after transformation {0}.".format(n_transformed_diff_cov))

print(">>> Relative difference before transformation:")
print("The relative difference is"
      " {0} %.".format(n_org_diff_cov / n_fix_cov * 100))
print("The relative difference is"
      " {0} %.".format(n_org_diff_cov / n_org_free_cov * 100))

print(">>> Relative difference after transformation:")
print("The relative difference is"
      " {0} %.".format(n_transformed_diff_cov / n_fix_cov * 100))
print("The relative difference is"
      " {0} %.".format(n_transformed_diff_cov / n_transformed_free_cov * 100))


# plt.show()
