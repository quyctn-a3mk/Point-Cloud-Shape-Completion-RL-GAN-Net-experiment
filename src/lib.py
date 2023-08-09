import numpy as np
import os
from multiprocessing import Pool
# import random

from external.python_plyfile.plyfile import PlyElement, PlyData

snc_synth_id_to_category = {
    '02691156': 'airplane',  '02773838': 'bag',        '02801938': 'basket',
    '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
    '02834778': 'bicycle',   '02843684': 'birdhouse',  '02871439': 'bookshelf',
    '02876657': 'bottle',    '02880940': 'bowl',       '02924116': 'bus',
    '02933112': 'cabinet',   '02747177': 'can',        '02942699': 'camera',
    '02954340': 'cap',       '02958343': 'car',        '03001627': 'chair',
    '03046257': 'clock',     '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table',     '04401088': 'telephone',  '02946921': 'tin_can',
    '04460130': 'tower',     '04468005': 'train',      '03085013': 'keyboard',
    '03261776': 'earphone',  '03325088': 'faucet',     '03337140': 'file',
    '03467517': 'guitar',    '03513137': 'helmet',     '03593526': 'jar',
    '03624134': 'knife',     '03636649': 'lamp',       '03642806': 'laptop',
    '03691459': 'speaker',   '03710193': 'mailbox',    '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano',     '03938244': 'pillow',     '03948459': 'pistol',
    '03991062': 'pot',       '04004475': 'printer',    '04074963': 'remote_control',
    '04090263': 'rifle',     '04099429': 'rocket',     '04225987': 'skateboard',
    '04256520': 'sofa',      '04330267': 'stove',      '04530566': 'vessel',
    '04554684': 'washer',    '02858304': 'boat',       '02992529': 'cellphone'
}

# def load_point_cloud(file_path, with_faces = False, with_color = False):
#     try:
#         ply_data = PlyData.read(file_path)
#         points = ply_data['vertex']
#         ret_val = [np.vstack([points['x'], points['y'], points['z']]).T]
#         if with_faces:
#             faces = np.vstack(ply_data['face']['vertex_indices'])
#             ret_val.append(faces)
#         if with_color:
#             r = np.vstack(ply_data['vertex']['red'])
#             g = np.vstack(ply_data['vertex']['green'])
#             b = np.vstack(ply_data['vertex']['blue'])
#             color = np.hstack((r, g, b))
#             ret_val.append(color)
#         return ret_val, ply_data
#     except:
#         return None, None

# def sparse_point_cloud(file_name, ratio_split = [50]):
#     point_cloud = load_point_cloud(file_name)
#     if not point_cloud:
#         return False
#     random.shuffle(point_cloud)
#     try:
#         for ratio in ratio_split:
#             sub_point_cloud = point_cloud[:ratio*len(ratio_split)//100]
#             PlyData.write(file_name)    
#         return True
#     except:
#         return False


def sparse_point_cloud_from_folder(root, contain, data, ratio_split = [50], ext = ['.ply']):
    data_ratio = {ratio: os.path.join(root, contain, data + '_' + str(ratio)) for ratio in ratio_split}
    for ratio in ratio_split:
        try:
            os.mkdir(data_ratio[ratio])
        except:
            print(f"Warning: {data_ratio[ratio]} already exist.")
    original_folder = os.path.join(root, contain, data)
    subclass_list = os.listdir(original_folder)
    ## loop for each sub class of point clouds
    for subclass_folder in subclass_list:
        if (not os.path.isdir(os.path.join(original_folder, subclass_folder))):
            continue
        else:
            print(f"Process subclass: {subclass_folder}")
        ## create folder contain ratio split
        for ratio in ratio_split:
            try:
                os.mkdir(os.path.join(data_ratio[ratio], subclass_folder))
            except:
                print(f"Warning: {os.path.join(data_ratio[ratio], subclass_folder)} already exist.")
        ## get list of point cloud files
        folder_full = os.path.join(original_folder, subclass_folder)
        file_list = [file if file.endswith(tuple(ext)) else '' for file in os.listdir(folder_full)]
        ## loop for each point cloud file
        for file_name in file_list:
            if not file_name:
                continue
            else:
                print(f"Process file: {file_name}")
            ## load file .ply
            file_path = os.path.join(folder_full, file_name)
            # point_cloud, ply_data = load_point_cloud(file_path)
            ply_data = PlyData.read(file_path)
            point_cloud = ply_data.elements
            if not point_cloud:
                return False
            for ratio in ratio_split:
                try:
                    file_ratio = os.path.join(data_ratio[ratio], subclass_folder, file_name)
                    sub_point_cloud = point_cloud
                    ## shufle for randomize choose by ratio_split: sort ply element list
                    np.random.shuffle(sub_point_cloud[0][:])
                    size = ratio*len(sub_point_cloud[0][:])//100
                    el = PlyElement.describe(sub_point_cloud[0][:size], 'vertex', comments=['vertices'])
                    ply_data.elements = [el]
                    ply_data.write(file_ratio)
                except:
                    print(f"Error: sparse_point_cloud at {file_ratio} failed.")

# root = '.'
# contain = 'data'
# data = 'shape_net_core_uniform_samples_2048'
# ratio_split = [80, 60, 40, 20]
# sparse_point_cloud_from_folder(root, contain, data, ratio_split)

def get_point_cloud_filepath_from_folder(parent_path, ext = ['.ply']):
    point_cloud_files_list = []
    subclass_list = os.listdir(parent_path)
    for subclass_folder in subclass_list:
        folder_contain = os.path.join(parent_path, subclass_folder)
        file_list = [file if file.endswith(tuple(ext)) else '' for file in os.listdir(folder_contain)]
        for file_name in file_list:
            if not file_name:
                continue
            point_cloud_files_list.append(os.path.join(folder_contain, file_name))
    return point_cloud_files_list

def list_point_cloud_filepath(root, contain, folder_contain = [], outFile_path = 'list_point_cloud_filepath.npy', ext = ['.ply']):
    point_cloud_files_list = []
    for folder in folder_contain:
        parent_path = os.path.join(root, contain, folder)
        read = get_point_cloud_filepath_from_folder(parent_path = parent_path, ext = ext)
        point_cloud_files_list.extend(read)
    np.save(outFile_path, point_cloud_files_list)

# folder_contain = ['shape_net_core_uniform_samples_2048',]
# list_point_cloud_filepath(root, contain, folder_contain, 'list_point_cloud_filepath.npy')