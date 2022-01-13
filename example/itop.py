# Start by importing all of the main libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sys
import pathlib
import open3d as o3d
import time
import torch.multiprocessing as multiprocessing

# Add the root directory so we can use the following imports
root_directory	=	str(pathlib.Path(__file__).parent.parent.resolve()).replace("\\","/")
sys.path.append(root_directory)


from src.v2v_model import V2VModel
from src.v2v_util_pointcloud import V2VVoxelization

from datasets.itop_side import ITOPDataset

model_path = root_directory + "/output/ITOP_side/model.pt"
model = torch.load(model_path);
model.eval()

device = None
if torch.cuda.is_available():
	device  =   torch.device('cuda')  
else:
	device  =   torch.device('cpu')

dtype = torch.float
model = model.to(device, dtype)

# Transform
cubic_size			=	3 #Size of cube (meters) around the center point to crop
v2v_voxelization	=	V2VVoxelization(cubic_size=cubic_size, original_size=100, augmentation=True)
voxelize_input		=	v2v_voxelization.voxelize
get_output			=	v2v_voxelization.evaluate
	


def thread_fn(input_queue, output_queue):
	while True:
		inputs = input_queue.get(block=True)		
		torch_input = inputs[0]
		reference_point = inputs[1]
		
		nn_input = torch_input.to(device, dtype)
		nn_output = model(nn_input[None, ...])
		
		nn_output = nn_output.cpu().detach().numpy()
		
		output = get_output(nn_output, [reference_point])[0]
		output_queue.put(output)


if __name__ == "__main__":
	input_queue = multiprocessing.Queue()
	output_queue = multiprocessing.Queue()

	nn_thread	=	multiprocessing.Process(target=thread_fn, args=(input_queue,output_queue))
	nn_thread.daemon = True
	nn_thread.start()
	
	
	data_dir		=	(root_directory + "/datasets/ITOP_side_processed/").replace("\\","/")
	test_data		=	 ITOPDataset(data_dir, 'test');
	

	data_pointcloud		=   o3d.geometry.PointCloud();
	output_pointcloud	=   o3d.geometry.PointCloud();
	temp_pointcloud		=   o3d.geometry.PointCloud();
	line_set = o3d.geometry.LineSet()


	'''
	joint_id_to_name = {
		0: 'Head',        8: 'Torso',
		1: 'Neck',        9: 'R Hip',
		2: 'R Shoulder',  10: 'L Hip',
		3: 'L Shoulder',  11: 'R Knee',
		4: 'R Elbow',     12: 'L Knee',
		5: 'L Elbow',     13: 'R Foot',
		6: 'R Hand',      14: 'L Foot',
		7: 'L Hand',
	}
	'''
	lines = [
		[0, 1],
		[2, 1],
		[3, 1],
		[2, 3],
		[4,2],
		[5,3],
		[6,4],
		[7,5],
		[8,2],
		[8,3],
		[9,8],
		[10,8],
		[10,9],
		[11,9],
		[12,10],
		[13,11],
		[14,12]
	]
	line_set.lines = o3d.utility.Vector2iVector(lines)

	visualizer = None
	visualizer2 = None

	# origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
	# visualizer.add_geometry(origin)
	last_time = 0
	i = -1;
	
	is_ready = True
	
	while True:
		current_time = time.time()
		if is_ready and (last_time + 0.016 < current_time):
			last_time = current_time
			if i < len(test_data):
				i += 1
				sample = test_data[i]
				
				points = sample['points']
				reference_point = sample['refpoint']
				voxelized_points = voxelize_input(points, reference_point)
				torch_input = torch.from_numpy(voxelized_points)
				
				input_queue.put([torch_input,reference_point])
				t2 = time.time()
				
				temp_pointcloud.points = o3d.utility.Vector3dVector(points);
				temp_pointcloud.points = temp_pointcloud.points
				# temp_pointcloud.points = temp_pointcloud.voxel_down_sample(0.05).points
				is_ready = False
			else:
				break
		
		if not output_queue.empty():
			output = output_queue.get_nowait()
			is_ready = True
			
			data_pointcloud.points = temp_pointcloud.points
			output_pointcloud.points = o3d.utility.Vector3dVector(output)
				
			line_set.points = output_pointcloud.points
			line_set.paint_uniform_color([0,0,1])
			# o3d.visualization.draw_geometries([line_set])
				
			# data_pointcloud.paint_uniform_color([0,0,1])
			output_pointcloud.paint_uniform_color([0,0,1])
			if visualizer is None:
				visualizer = o3d.visualization.Visualizer()
				visualizer.create_window()
				options = visualizer.get_render_option()
				options.point_size = 3.0
				visualizer.add_geometry(data_pointcloud)
			else:
				visualizer.update_geometry(data_pointcloud)
				
				
			if visualizer2 is None:
				visualizer2 = o3d.visualization.Visualizer()
				visualizer2.create_window()
				options = visualizer2.get_render_option()
				options.point_size = 3.0
				visualizer2.add_geometry(line_set)
			else:
				visualizer2.update_geometry(line_set)
		
		if visualizer is not None:
			visualizer.poll_events()
			visualizer.update_renderer()
		if visualizer2 is not None:
			visualizer2.poll_events()
			visualizer2.update_renderer()





				
				


				
				
