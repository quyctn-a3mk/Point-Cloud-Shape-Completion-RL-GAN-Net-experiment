import os
import shutil
from random import randrange

from . utils import load_all_point_clouds_under_folder

# ratio = 0.85

# org_root = "../data/shape_net_core_uniform_samples_2048/"
# folder_list = os.listdir(org_root)

# new_root = org_root + "_split"
# train_root = new_root + "/train"
# test_root = new_root + "/test"

# os.mkdir(new_root)
# os.mkdir(train_root)
# os.mkdir(test_root)

# for folder in folder_list:
#     if (not os.path.isdir(os.path.join(org_root, folder))):
#         continue
#     os.mkdir(os.path.join(train_root, folder))
#     os.mkdir(os.path.join(test_root, folder))
    
#     ply_list = os.listdir(os.path.join(org_root, folder))
    
#     # Calculate number of tests
#     test_amount = round(len(ply_list) * (1-ratio))
#     selection_map = [0 for i in range(len(ply_list))]
    
#     # Select 15% test data
#     selected = 0
#     while (selected != test_amount):
#         tmp = randrange(len(ply_list))
#         if (selection_map[tmp] == 0):
#             selected += 1
#             selection_map[tmp] = 1
    
#     for i,file in enumerate(ply_list):
#         source = os.path.join(org_root, folder, file)
#         if (selection_map[i] == 0):
#             dest = os.path.join(train_root, folder, file)
#         else:
#             dest = os.path.join(test_root, folder, file)
#         shutil.copyfile(source, dest)

root = "../data/"
data_full = root + "/shape_net_core_uniform_samples_2048/"
folder_list = os.listdir(org_root)

ratio_split = [80, 60, 40, 20]
ext = '.ply'
for ratio in ratio_split:
    data_ratio = root + '_' + str(ratio)
    os.mkdir(data_ratio)
    for folder in folder_list:
        if (not os.path.isdir(os.path.join(data_full, folder))):
            continue
        os.mkdir(os.path.join(data_ratio, folder))
        class_dir = os.path.join(data_full , folder)
        ply_list = [file if file.endswith(ext) for file in os.listdir(class_dir)]


        # all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=20, file_ending='.ply', verbose=True)
        # for pcloud in all_pc_data.point_clouds:
            # seed_idx = int(np.random.rand() * 2048 * 0.8)

def load_point_clouds(file_names, ext = '.ply', numThread = 20):
    # ply_list = [file if file.endswith(ext) for file in os.listdir(class_dir)]
    
    
    
    pool = Pool(numThread)


    for i, data in enumerate(pool.imap(loader, file_names)):
        pclouds[i, :, :], model_names[i], class_ids[i] = data

    pool.close()
    pool.join()

    if len(np.unique(model_names)) != len(pclouds):
        warnings.warn('Point clouds with the same model name were loaded.')

    if verbose:
        print('{0} pclouds were loaded. They belong in {1} shape-classes.'.format(len(pclouds), len(np.unique(class_ids))))

    return pclouds, model_names, class_ids



list_new_data = []
for i in tqdm.trange(len(list_point_clouds)):
    points = PyntCloud.from_file(list_point_clouds[i])
    points = np.array(points.points)
    seed_idx = int(np.random.rand() * 2048 * 0.8)
    points_removed = np.concatenate((points[:seed_idx, :], points[seed_idx + int(0.2*2048):, :]), axis=0)
    list_new_data.append(points_removed)

np.save('list_point_noisy.npy', list_new_data)