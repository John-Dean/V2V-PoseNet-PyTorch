# Start by importing all of the main libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import os
import sys
import pathlib
import shutil
import open3d as o3d
from datetime import datetime

# Add the root directory so we can use the following imports
root_directory	=	str(pathlib.Path(__file__).parent.parent.parent.resolve()).replace("\\","/")
sys.path.append(root_directory)

from lib.solver import train_epoch, val_epoch, test_epoch
from lib.sampler import ChunkSampler
from src.v2v_model import V2VModel
# from src.v2v_util import V2VVoxelization
from src.v2v_util_pointcloud import V2VVoxelization
from datasets.msra_hand import MSRAHandDataset

device = None
if torch.cuda.is_available():
	device  =   torch.device('cuda')  
else:
	device  =   torch.device('cpu')

dtype = torch.float
start_epoch = 0

# Configuration
epochs_num = 15
batch_size = 12
checkpoint_dir	=	(root_directory +  "/checkpoint/").replace("\\","/")
output_dir		=	(root_directory + "/output/").replace("\\","/")

# Dataset info
data_dir		=	(root_directory + "/datasets/cvpr15_MSRAHandGestureDB/").replace("\\","/")
center_dir		=	(root_directory + "/datasets/msra_center/").replace("\\","/")
keypoints_num	=	21
test_subject_id	=	3
cubic_size		=	200

# Transform
voxelization_train	=	V2VVoxelization(cubic_size=cubic_size, augmentation=True)
voxelization_val	=	V2VVoxelization(cubic_size=cubic_size, augmentation=False)
voxelize_input		=	voxelization_train.voxelize
evaluate_keypoints	=	voxelization_train.evaluate

# Training functions
def transform_train(sample):
	points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
	assert(keypoints.shape[0] == keypoints_num)
	input, heatmap = voxelization_train({'points': points, 'keypoints': keypoints, 'refpoint': refpoint})
	return (torch.from_numpy(input), torch.from_numpy(heatmap))

def transform_val(sample):
	points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
	assert(keypoints.shape[0] == keypoints_num)
	input, heatmap = voxelization_val({'points': points, 'keypoints': keypoints, 'refpoint': refpoint})
	return (torch.from_numpy(input), torch.from_numpy(heatmap))


# Testing functions
def transform_test(sample):
	points, refpoint = sample['points'], sample['refpoint']
	input = voxelize_input(points, refpoint)
	return torch.from_numpy(input), torch.from_numpy(refpoint.reshape((1, -1)))

def transform_output(heatmaps, refpoints):
	keypoints = evaluate_keypoints(heatmaps, refpoints)
	return keypoints

class BatchResultCollector():
	def __init__(self, samples_num, transform_output):
		self.samples_num = samples_num
		self.transform_output = transform_output
		self.keypoints = None
		self.idx = 0
	
	def __call__(self, data_batch):
		inputs_batch, outputs_batch, extra_batch = data_batch
		outputs_batch = outputs_batch.cpu().numpy()
		refpoints_batch = extra_batch.cpu().numpy()

		keypoints_batch = self.transform_output(outputs_batch, refpoints_batch)

		if self.keypoints is None:
			# Initialize keypoints until dimensions available now
			self.keypoints = np.zeros((self.samples_num, *keypoints_batch.shape[1:]))

		batch_size = keypoints_batch.shape[0] 
		self.keypoints[self.idx:self.idx+batch_size] = keypoints_batch
		self.idx += batch_size

	def get_result(self):
		return self.keypoints

def save_keypoints(filename, keypoints):
	# Reshape one sample keypoints into one line
	keypoints = keypoints.reshape(keypoints.shape[0], -1)
	np.savetxt(filename, keypoints, fmt='%0.4f')


if __name__ ==  '__main__':
	print('==> Starting')
	
	# Dataset and loader
	train_set		=	MSRAHandDataset(data_dir, center_dir, 'train',	test_subject_id, transform_train)
	train_loader	=	torch.utils.data.DataLoader(train_set,	batch_size=batch_size, shuffle=True, num_workers=6)

	# No separate validation dataset, just use test dataset instead
	val_set			=	MSRAHandDataset(data_dir, center_dir, 'test',	test_subject_id, transform_val)
	val_loader		=	torch.utils.data.DataLoader(val_set,	batch_size=batch_size, shuffle=False, num_workers=6)

	print('==> Constructing model ..')
	model = V2VModel(input_channels=1, output_channels=keypoints_num)

	model = model.to(device, dtype)
	if device == torch.device('cuda'):
		torch.backends.cudnn.enabled = True
		cudnn.benchmark = True
		print('cudnn.enabled: ', torch.backends.cudnn.enabled)

	criterion = nn.MSELoss()

	optimizer = optim.Adam(model.parameters())
	#optimizer = optim.RMSprop(net.parameters(), lr=2.5e-4)
	
	# We try to resume
	if os.path.isdir(checkpoint_dir):
		# Load checkpoint
		epoch = 0
		searching = True
		
		while searching:
			checkpoint_file = (checkpoint_dir + 'epoch'+str(epoch)+'.pth')
			if os.path.isfile(checkpoint_file):
				epoch+=1
			else:
				searching = False
				epoch-=1
	
		if epoch >= 0:
			print('==> Resuming from checkpoint after epoch {} ..'.format(epoch))
			
			checkpoint_file = (checkpoint_dir + 'epoch'+str(epoch)+'.pth')
			
			assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
			assert os.path.isfile(checkpoint_file), 'Error: no checkpoint file of epoch {}'.format(epoch)

			checkpoint = torch.load(checkpoint_file)
			model.load_state_dict(checkpoint['model_state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			start_epoch = checkpoint['epoch'] + 1
	
	print('==> Training ..')
	for epoch in range(start_epoch, epochs_num):
		print('Epoch: {}'.format(epoch))
		train_epoch(model, criterion, optimizer, train_loader, device=device, dtype=dtype)
		val_epoch(model, criterion, val_loader, device=device, dtype=dtype)
		if not os.path.exists(checkpoint_dir):
			os.mkdir(checkpoint_dir)
		
		checkpoint_file = os.path.join(checkpoint_dir, 'epoch'+str(epoch)+'.pth')
		checkpoint = {
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'epoch': epoch
		}
		torch.save(checkpoint, checkpoint_file)

	print('==> Testing ..')
	print('Test on test dataset ..')
	test_set = MSRAHandDataset(data_dir, center_dir, 'test', test_subject_id, transform_test)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=6)
	test_res_collector = BatchResultCollector(len(test_set), transform_output)

	test_epoch(model, test_loader, test_res_collector, device, dtype)
	keypoints_test = test_res_collector.get_result()


	print('Fit on train dataset ..')
	fit_set = MSRAHandDataset(data_dir, center_dir, 'train', test_subject_id, transform_test)	
	fit_loader = torch.utils.data.DataLoader(fit_set, batch_size=batch_size, shuffle=False, num_workers=6)
	fit_res_collector = BatchResultCollector(len(fit_set), transform_output)

	test_epoch(model, fit_loader, fit_res_collector, device, dtype)
	keypoints_fit = fit_res_collector.get_result()
	
	
	print('Gather test keypoints');
	test_set = MSRAHandDataset(data_dir, center_dir, 'test', test_subject_id)
	test_joints = []
	
	for i in range(len(test_set)):
		record = test_set[i]
		test_joints.append(record['joints'])
	
	
	print('Gather fit keypoints');
	fit_set = MSRAHandDataset(data_dir, center_dir, 'train', test_subject_id)	
	fit_joins = []
	
	for i in range(len(fit_set)):
		record = fit_set[i]
		fit_joins.append(record['joints'])
	
	
	print('Saving model ..')
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	# Add timestamp to the output
	output_dir = output_dir + datetime.today().strftime('%Y-%m-%d_%H-%M-%S') + "/"
	
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	
	save_keypoints(output_dir + "./train_gt.txt", np.array(fit_joins))
	save_keypoints(output_dir + "./test_gt.txt", np.array(test_joints))
	save_keypoints(output_dir + "/test_res.txt", keypoints_test)
	save_keypoints(output_dir + "/fit_res.txt", keypoints_fit)
	
	
	accuracy_file = str(pathlib.Path(__file__).parent.parent.resolve()).replace("\\","/")
	
	accuracy_file += "/accuracy_graph.py"
	
	shutil.copyfile(accuracy_file, output_dir + "/accuracy_graph.py")
	
	shutil.move(checkpoint_dir,output_dir)
	torch.save(model, output_dir + "/model.pt")
	
	print('All done ..')
	
	
