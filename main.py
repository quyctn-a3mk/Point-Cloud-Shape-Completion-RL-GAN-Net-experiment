import os
from external.python_plyfile.plyfile import PlyElement, PlyData
from src.lib import sparse_point_cloud_from_folder, list_point_cloud_filepath

def main():
    root = '.'
    contain = 'data'
    data = 'shape_net_core_uniform_samples_2048'
    ratio_split = [80, 60, 40, 20]
    sparse_point_cloud_from_folder(root, contain, data, ratio_split)
    folder_contain = ['shape_net_core_uniform_samples_2048',]
    outFile_path = os.path.join(root, contain, 'filter', 'list_point_cloud_filepath.npy')
    list_point_cloud_filepath(root, contain, folder_contain, outFile_path, ['.ply'])    

if __name__ == '__main__':
    main()