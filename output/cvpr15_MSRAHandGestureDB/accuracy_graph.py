import sys
import pathlib
import glob
import os
sys.path.append('../../')

root_directory	=	str(pathlib.Path(__file__).parent.parent.parent.resolve()).replace("\\","/")
sys.path.append(root_directory)

import numpy as np
import matplotlib.pyplot as plt
from lib.accuracy import *
from vis.plot import *



latest_file = str(pathlib.Path(__file__).parent.resolve()).replace("\\","/")
pred_file = latest_file + "/test_res.txt"
gt_file = latest_file + "/test_gt.txt"



gt = np.loadtxt(gt_file)
gt = gt.reshape(gt.shape[0], -1, 3)

pred = np.loadtxt(pred_file)
pred = pred.reshape(pred.shape[0], -1, 3)

print('gt: ', gt.shape)
print('pred: ', pred.shape)

keypoints_num = pred.shape[1]
names = ['joint'+str(i+1) for i in range(keypoints_num)]


dist, acc = compute_dist_acc_wrapper(pred, gt, max_dist=100, num=100)

fig, ax = plt.subplots()
plot_acc(ax, dist, acc, names)
fig.savefig('accuracy-plot.png')
plt.show()


mean_err = compute_mean_err(pred, gt)
fig, ax = plt.subplots()
plot_mean_err(ax, mean_err, names)
fig.savefig('mean-error-plot.png')
plt.show()


print('mean_err: {}'.format(mean_err))
mean_err_all = compute_mean_err(pred.reshape((-1, 1, 3)), gt.reshape((-1, 1,3)))
print('mean_err_all: ', mean_err_all)
