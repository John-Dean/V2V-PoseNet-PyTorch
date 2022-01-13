import numpy as np
import open3d as o3d

def warp2continuous(coord, refpoint, cubic_size, cropped_size):
    '''
    Map coordinates in set [0, 1, .., cropped_size-1] to original range [-cubic_size/2+refpoint, cubic_size/2 + refpoint]
    '''
    min_normalized = -1
    max_normalized = 1

    scale = (max_normalized - min_normalized) / cropped_size
    coord = coord * scale + min_normalized  # -> [-1, 1]

    coord = coord * cubic_size / 2 + refpoint

    return coord

def extract_coord_from_output(output, center=True):
    '''
    output: shape (batch, jointNum, volumeSize, volumeSize, volumeSize)
    center: if True, add 0.5, default is true
    return: shape (batch, jointNum, 3)
    '''
    assert(len(output.shape) >= 3)
    vsize = output.shape[-3:]

    output_rs = output.reshape(-1, np.prod(vsize))
    max_index = np.unravel_index(np.argmax(output_rs, axis=1), vsize)
    max_index = np.array(max_index).T
    
    xyz_output = max_index.reshape([*output.shape[:-3], 3])

    # Note discrete coord can represents real range [coord, coord+1), see function scattering() 
    # So, move coord to range center for better fittness
    if center: xyz_output = xyz_output + 0.5

    return xyz_output

def scattering_pointcloud(pointcloud, cropped_size):
     #Trim off any extreme edge points (as these will round up into an non-existant voxel)
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(
            0,
            0,
            0
        ),
        max_bound=(
            cropped_size-0.01,
            cropped_size-0.01,
            cropped_size-0.01
        )
    )
    pointcloud  =   pointcloud.crop(bounding_box) 
    
    points = np.trunc(np.asarray(pointcloud.points)).astype(int)
    
    #Convert to a output 3d array
    output = np.zeros((cropped_size, cropped_size, cropped_size))
    for i in range(len(points)):
        point = points[i]
        output[point[0]][point[1]][point[2]] = 1.0
        
    return output

def generate_coord_pointcloud(pointcloud, refpoint, new_size, angle, trans, sizes):
    cubic_size, cropped_size, original_size = sizes
    
    # normalize
    pointcloud.translate(-refpoint) #Translate the point cloud so refpoint is in the center
    pointcloud.scale(1 / (cubic_size/2), [0,0,0])  # Scale them between [-1, 1]
    
    #This seems to break stuff, leave off!
    # # (Additional) Crop the pointcloud around [-1, 1]
    # bounding_box = o3d.geometry.AxisAlignedBoundingBox(
    #     min_bound=(
    #         -1,
    #         -1,
    #        -1
    #     ),
    #     max_bound=(
    #         1,
    #         1,
    #        1
    #     )
    # )
    # pointcloud  =   pointcloud.crop(bounding_box)
    
    #discretize
    pointcloud.translate([1,1,1]) #Translate so (0,0,0) is in the center, and now scaled from [0,2]
    pointcloud.scale(cropped_size / 2, [0,0,0]) # Scale between [0, cropped_size]
    pointcloud.translate([
        (original_size / 2 - cropped_size / 2),
        (original_size / 2 - cropped_size / 2),
        (original_size / 2 - cropped_size / 2)
    ])
    
    # resize around original volume center
    resize_scale = new_size / 100
    if new_size < 100:
        pointcloud.scale(resize_scale, [0,0,0])
        pointcloud.translate([
            (original_size / 2) * (1 - resize_scale),
            (original_size / 2) * (1 - resize_scale),
            (original_size / 2) * (1 - resize_scale),
        ])
    elif new_size > 100:
        pointcloud.scale(resize_scale, [0,0,0])
        pointcloud.translate([
            -(original_size / 2) * (resize_scale - 1),
            -(original_size / 2) * (resize_scale - 1),
            -(original_size / 2) * (resize_scale - 1),
        ])
    else:
        # new_size = 100 if it is in test mode
        pass
    
    # rotation
    if angle != 0:
        pointcloud.translate([
            -(original_size / 2),
            -(original_size / 2),
            0
        ])
        pointcloud.rotate([
            [np.cos(angle)  , -np.sin(angle)    , 0],
            [np.sin(angle)  , np.cos(angle)     , 0],
            [0              , 0                 , 1],
        ], [0,0,0])   
        pointcloud.translate([
            (original_size / 2),
            (original_size / 2),
            0
        ])
    
    # # translation
    if not isinstance(trans, (list, tuple, np.ndarray)):
        trans = np.array([trans, trans, trans])
    
    pointcloud.translate(-trans)
    
    # #Trim off any extreme edge points (as these will round up into an non-existant voxel)
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(
            0,
            0,
            0
        ),
        max_bound=(
            cropped_size,
            cropped_size,
            cropped_size
        )
    )
    pointcloud  =   pointcloud.crop(bounding_box) 
    
    return np.asarray(pointcloud.points)

def generate_cubic_input(pointcloud, refpoint, new_size, angle, trans, sizes):
    _, cropped_size, _ = sizes
    
    generate_coord_pointcloud(pointcloud, refpoint, new_size, angle, trans, sizes)
    
    # scattering
    cubic = scattering_pointcloud(pointcloud, cropped_size) 
    
    
    # points = np.asarray(pointcloud.points);
    # coord_old = generate_coord(points, refpoint, new_size, angle, trans, sizes)
    # cubic_old = scattering(coord_old, cropped_size)
    # print("are equal?")
    # print(np.array_equal(cubic, cubic_old))
    # input("Pause to confirm")


    return cubic

def generate_heatmap_gt(pointcloud, refpoint, new_size, angle, trans, sizes, d3outputs, pool_factor, std):    
    number_of_keypoints = len(pointcloud.points)    
    
    _, cropped_size, _ = sizes
    d3output_x, d3output_y, d3output_z = d3outputs

    # coord = generate_coord(keypoints, refpoint, new_size, angle, trans, sizes)  # [0, cropped_size]
    coord = generate_coord_pointcloud(pointcloud, refpoint, new_size, angle, trans, sizes)
    
    coord /= pool_factor  # [0, cropped_size/pool_factor]

    # heatmap generation
    output_size = int(cropped_size / pool_factor)
    heatmap = np.zeros((number_of_keypoints, output_size, output_size, output_size))

    # use center of cell
    center_offset = 0.5

    for i in range(coord.shape[0]):
        xi, yi, zi= coord[i]
        heatmap[i] = np.exp(-(np.power((d3output_x+center_offset-xi)/std, 2)/2 + \
            np.power((d3output_y+center_offset-yi)/std, 2)/2 + \
            np.power((d3output_z+center_offset-zi)/std, 2)/2))

    return heatmap


class V2VVoxelization(object):
    def __init__(self, cubic_size, augmentation=True, cropped_size = 88, original_size = 96, pool_factor = 2, std = 1.7):
        self.cubic_size = cubic_size
        self.cropped_size, self.original_size = cropped_size, original_size
        self.sizes = (self.cubic_size, self.cropped_size, self.original_size)
        self.pool_factor = pool_factor
        self.std = std
        self.augmentation = augmentation

        output_size = int(self.cropped_size / self.pool_factor)
        # Note, range(size) and indexing = 'ij'
        self.d3outputs = np.meshgrid(np.arange(output_size), np.arange(output_size), np.arange(output_size), indexing='ij')

    def __call__(self, sample):
        points, keypoints, refpoint = sample['points'], sample['keypoints'], sample['refpoint']
        
        ## Setup o3d pointclouds
        data_pointcloud =   o3d.geometry.PointCloud();
        data_pointcloud.points = o3d.utility.Vector3dVector(points)
        
        keypoints_pointcloud =   o3d.geometry.PointCloud();
        keypoints_pointcloud.points = o3d.utility.Vector3dVector(keypoints)
        
        
        ## Augmentations
        # Resize
        new_size = np.random.rand() * 40 + 80

        # Rotation
        angle = np.random.rand() * 80/180*np.pi - 40/180*np.pi

        # Translation
        trans = np.random.rand(3) * (self.original_size-self.cropped_size)
        
        if not self.augmentation:
            new_size = 100
            angle = 0
            trans = self.original_size/2 - self.cropped_size/2

        #Generate outputs
        nn_input = generate_cubic_input(data_pointcloud, refpoint, new_size, angle, trans, self.sizes)
        heatmap = generate_heatmap_gt(keypoints_pointcloud, refpoint, new_size, angle, trans, self.sizes, self.d3outputs, self.pool_factor, self.std)

        return nn_input.reshape((1, *nn_input.shape)), heatmap

    def voxelize(self, points, refpoint):
        new_size, angle, trans = 100, 0, self.original_size/2 - self.cropped_size/2
        
        ## Setup pointclouds
        data_pointcloud =   o3d.geometry.PointCloud();
        data_pointcloud.points = o3d.utility.Vector3dVector(points)
        
        nn_input = generate_cubic_input(data_pointcloud, refpoint, new_size, angle, trans, self.sizes)
        return nn_input.reshape((1, *nn_input.shape))

    def generate_heatmap(self, keypoints, refpoint):
        new_size, angle, trans = 100, 0, self.original_size/2 - self.cropped_size/2
        
        ## Setup pointclouds
        keypoints_pointcloud =   o3d.geometry.PointCloud();
        keypoints_pointcloud.points = o3d.utility.Vector3dVector(keypoints)
        
        heatmap = generate_heatmap_gt(keypoints_pointcloud, refpoint, new_size, angle, trans, self.sizes, self.d3outputs, self.pool_factor, self.std)
        return heatmap

    def evaluate(self, heatmaps, refpoints):        
        # heatmaps is an array of length batch_size containing keypoints number of output_size size 3d arrays
        # repoints is an array of length batch_size containing data
        
        # This line of code takes the given heatmaps and convert to real points
        # i.e. if the hotest part of the heat map is at [5,5,5] this returns [5,5,5]
        # Return is an array of length batch_size containing keypoints number of [x,y,z] coordinates
        coords = extract_coord_from_output(heatmaps)
        
        # Upscale the output by the pool factor (to match the "cropped_size" input)
        coords *= self.pool_factor
        
        # This then converts back to the original coordinate space
        keypoints = warp2continuous(coords, refpoints, self.cubic_size, self.cropped_size)
        
        return keypoints
