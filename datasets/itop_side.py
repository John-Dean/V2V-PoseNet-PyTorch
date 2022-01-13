import numpy as np
import os
from torch.utils.data import Dataset

class ITOPDataset(Dataset):
	def __init__(self, dataset_directory, mode, transform=None):
		self.img_width = 320
		self.img_height = 240
		
		self.joint_num = 15 # Number of joins

		self.dataset_directory = dataset_directory    # Directory of the dataset
		self.mode = mode # Either test or train
		self.transform = transform # Transform function
		
		self.location = None
		
		self.dataset = None
		self.num_samples = 5
		
		self._load()
	
	def __getitem__(self, index):
		#[unique_id, person_number, frame_number, pointcloud, keypoint, center]
		dataset_record = np.load(self.location + str(index) + ".npy", allow_pickle=True)
		
		sample = {
            'name': dataset_record[0],
            'points': dataset_record[3],
            'joints': dataset_record[4],
            'refpoint': dataset_record[5]
        }
		
		if self.transform:
			sample = self.transform(sample)

		return sample


	def __len__(self):
		return self.num_samples


	def _load(self):
		
		if self.mode == 'test':
			self.location	=	self.dataset_directory + 'test/';
		else:
			self.location	=	self.dataset_directory + 'train/';
		
		files = next(os.walk(self.location))[2]
		
		self.num_samples = len(files)
