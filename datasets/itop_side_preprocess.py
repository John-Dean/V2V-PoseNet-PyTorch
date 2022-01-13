import h5py
import numpy as np
import os
from torch.utils.data import Dataset

class ITOPDataset(Dataset):
	def __init__(self, root, center_dir, mode, transform=None):
		self.img_width = 320
		self.img_height = 240
		
		self.joint_num = 15 # Number of joins

		self.root = root    # Directory of the dataset
		self.center_dir = center_dir # Directory of the centers
		self.mode = mode # Either test or train
		self.transform = transform # Transform function
		
		self.dataset = None
		self.num_samples = None
		
		self._load()
	
	def __getitem__(self, index):
		dataset_record = self.dataset[index]
		
		sample = {
            'name': dataset_record.unique_id,
            'points': dataset_record.pointcloud,
            'joints': dataset_record.keypoints,
            'refpoint': dataset_record.center
        }
		
		if self.transform:
			sample = self.transform(sample)

		return sample


	def __len__(self):
		return self.num_samples


	def _load(self):
		dataset = []	
		
		pointcloud_file_location = ""
		labels_file_location = ""
		center_path = ""
		
		if self.mode == 'test':
			pointcloud_file_location	=	self.root + 'ITOP_side_test_point_cloud.h5';
			labels_file_location	=	self.root + 'ITOP_side_test_labels.h5';
			center_path	=	self.center_dir + 'center_test.txt';
		else:
			pointcloud_file_location	=	self.root + 'ITOP_side_train_point_cloud.h5';
			labels_file_location	=	self.root + 'ITOP_side_train_labels.h5';
			center_path	=	self.center_dir + 'center_train.txt';
		
		centers = np.genfromtxt(center_path)
		
		pointcloud_file_data = h5py.File(pointcloud_file_location, 'r')
		data, ids = pointcloud_file_data.get('data'), pointcloud_file_data.get('id')
		
		pointclouds, ids = np.asarray(data), np.asarray(ids);
		
		labels_file_data = h5py.File(labels_file_location, 'r')
		keypoints, valid_keypoints = labels_file_data.get('real_world_coordinates'), labels_file_data.get('is_valid')
		
		for i in range(len(ids)):
			unique_id = str(ids[i])
			person_number = int(unique_id[2:4])
			frame_number = int(unique_id[-6:-1])
			
			unique_id = str(person_number) + "_" + str(frame_number)
			
			pointcloud = pointclouds[i]
			keypoint = keypoints[i]
			is_valid = valid_keypoints[i]
			center = centers[i]
			
			if is_valid > 0:
				if not np.isnan(center[0]):
					dataset.append([unique_id, person_number, frame_number, pointcloud, keypoint, center])
		
		self.dataset = dataset
		self.num_samples = len(dataset)


import pathlib
root_directory	=	str(pathlib.Path(__file__).parent.parent.resolve()).replace("\\","/")
data_dir		=	(root_directory + "/datasets/ITOP/").replace("\\","/")
center_dir		=	(root_directory + "/datasets/ITOP_side_center/").replace("\\","/")
output_dir		=	(root_directory + "/datasets/ITOP_processed/").replace("\\","/")


if not os.path.exists(output_dir):
	os.mkdir(output_dir)
if not os.path.exists(output_dir + "test/"):
	os.mkdir(output_dir + "test/")
if not os.path.exists(output_dir + "train/"):
	os.mkdir(output_dir + "train/")

test = ITOPDataset(data_dir, center_dir, 'test')
train = ITOPDataset(data_dir, center_dir, 'train')

dataset_test = test.dataset;
dataset_train = train.dataset;

for i in range(len(dataset_test)):
	record = dataset_test[i]
	filename = "test/" + str(i)
	
	np.save(output_dir + filename, np.array(record, dtype=object))

for i in range(len(dataset_train)):
	record = dataset_train[i]
	filename = "train/" + str(i)
	
	np.save(output_dir + filename, np.array(record, dtype=object))

print("Done")
