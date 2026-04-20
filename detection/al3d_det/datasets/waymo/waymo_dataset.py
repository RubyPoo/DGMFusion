import os
import pickle
import io
import copy
import re

import numpy as np
import cv2
#from petrel_client.client import Client
from al3d_utils import common_utils
from al3d_utils.ops.roiaware_pool3d import roiaware_pool3d_utils
from al3d_utils.aws_utils import list_oss_dir, oss_exist

from al3d_det.datasets.dataset import DatasetTemplate
from al3d_det.datasets.augmentor.data_augmentor import DataAugmentor
from al3d_det.datasets.augmentor.test_time_augmentor import TestTimeAugmentor


class WaymoInferenceDataset(DatasetTemplate):
    """
    The Dataset class for Inference on Waymo
    """
    def __init__(self, dataset_cfg, class_names, data_infos, point_list, training=False, logger=None) -> None:
        super().__init__(dataset_cfg, class_names, training, logger)
        self.data_infos = data_infos
        self.point_list = point_list
        self.init_infos()

    def init_infos(self):
        self.infos = self.data_infos

    def get_infos_and_points(self, idx_list):
        infos, points = [], []
        for i in idx_list:
            infos.append(self.infos[i])
            points.append(self.point_list[i])
        return infos, points


class WaymoTrainingDataset(DatasetTemplate):
    """
    The Dataset class for Training on Waymo (from File System)
    """

    def __init__(self, dataset_cfg, class_names, root_path, training=True, logger=None) -> None:
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        self.data_path = self.root_path + '/' + dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = os.path.join(self.root_path, 'ImageSets', self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.init_infos()

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names,
            training=self.training, root_path=self.root_path,
            logger=self.logger
        )
        self.split = split
        split_dir = os.path.join(self.root_path, 'ImageSets', self.split+'.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []
        self.init_infos()

    def init_infos(self):
        self.logger.info('Loading Waymo dataset')
        waymo_infos = []
        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            info_path = os.path.join(self.data_path, sequence_name, ('%s.pkl' % sequence_name))
            
            info_path = self.check_sequence_name_with_all_version(info_path)
            
            if not os.path.exists(info_path):
                    num_skipped_infos += 1
                    continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
            
            waymo_infos.extend(infos)

        self.infos.extend(waymo_infos[:])
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos)))
        if self.dataset_cfg.SAMPLED_INTERVAL[self.mode] > 1:
            sampled_waymo_infos = []
            for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[self.mode]):
                sampled_waymo_infos.append(self.infos[k])
            self.infos = sampled_waymo_infos
            self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.infos))

    def check_sequence_name_with_all_version(self, seq_file):
        if 's3' in self.root_path:
            if '_with_camera_labels' not in seq_file and\
                not oss_exist(self.data_path, seq_file, self.oss_data_list):
                seq_file = seq_file[:-9] + '_with_camera_labels.tfrecord'
            if '_with_camera_labels' in seq_file and\
                not oss_exist(self.data_path, seq_file, self.oss_data_list):
                seq_file = seq_file.replace('_with_camera_labels', '')
        else:
            if '_with_camera_labels' not in seq_file and not os.path.exists(seq_file):
                seq_file = seq_file[:-9] + '_with_camera_labels.tfrecord'
            if '_with_camera_labels' in seq_file and not os.path.exists(seq_file):
                seq_file = seq_file.replace('_with_camera_labels', '')

        return seq_file

    def get_infos_and_points(self, idx_list):
        infos, points = [], []
        for i in idx_list:
            lidar_path = self.infos[i]["lidar_path"]
            lidar_path = str(lidar_path).replace('/cpfs2/user/matao/workspace/3dal-toolchain-v2/detection/data', '../data')
            current_point = np.load(lidar_path)
            
            infos.append(self.infos[i])
            points.append(current_point)

        return infos, points

    def get_images_and_params(self, current_idx, idx_list):
        imgs_dict = {
            'images': {},
            'extrinsic': {},
            'intrinsic': {},
            'image_shape': {}
        }

        # 修复：正确检查idx_list是否为空
        if len(idx_list) == 0:
            return imgs_dict

        for i in idx_list:
            if not self.load_multi_images:
                if i != current_idx: 
                    continue
            
            img_infos = self.infos[i]['image']
            sequence_name = self.infos[i]['sequence_name']
            sample_idx = self.infos[i]['sample_idx']
            
            # 构建序列目录路径
            sequence_dir = os.path.join(self.data_path, sequence_name)
            
            # 首先检查图像目录是否存在
            image_dirs_exist = False
            for j in range(5):
                img_dir = os.path.join(sequence_dir, f'image_{j}')
                if os.path.exists(img_dir):
                    image_dirs_exist = True
                    break
            
            if not image_dirs_exist:
                self.logger.warning(f"No image directories found in sequence: {sequence_dir}")
                # 继续处理，但跳过这个序列的图像加载
                continue

            for key in img_infos.keys():
                if 'path' not in key: 
                    continue
                    
                original_img_path = img_infos[key]
                
                # 从原始路径中提取图像文件名
                img_filename = os.path.basename(original_img_path)
                
                for j in range(5):
                    # 构建新的图像路径
                    img_dir = os.path.join(sequence_dir, f'image_{j}')
                    
                    # 尝试多种可能的文件名格式
                    possible_filenames = [
                        img_filename,  # 原始文件名
                        f"{sample_idx:04d}.png",  # 样本索引格式
                        f"{sample_idx:04d}.jpg",  # jpg格式
                        f"{sample_idx}.png",  # 不带前导零
                        f"{sample_idx}.jpg",  # jpg格式不带前导零
                    ]
                    
                    img_path = None
                    for filename in possible_filenames:
                        test_path = os.path.join(img_dir, filename)
                        if os.path.exists(test_path):
                            img_path = test_path
                            break
                    
                    if img_path is None:
                        # 如果仍然找不到，记录调试信息
                        if j == 0:  # 只对第一个相机记录详细信息
                            self.logger.debug(f"Image file not found in {img_dir}. Tried: {possible_filenames}")
                        continue
                    
                    try:
                        image = cv2.imread(img_path)
                        if image is None:
                            self.logger.warning(f"Failed to load image: {img_path}")
                            continue
                            
                        # normalize images
                        image = image.astype(np.float32)
                        image /= 255.0
                        cam_name = 'camera_%s' % str(j)
                        
                        # resize image
                        if self.image_scale != 1:
                            new_shape = [int(image.shape[1]*self.image_scale), int(image.shape[0]*self.image_scale)]
                            image = cv2.resize(image, new_shape)
                            img_infos['image_shape_%d' % j] = new_shape[::-1]
                        
                        if cam_name not in imgs_dict['images']:
                            imgs_dict['images'][cam_name] = []
                        imgs_dict['images'][cam_name].append(image)
                        
                    except Exception as e:
                        self.logger.warning(f"Error loading image {img_path}: {e}")
                        continue

        # 如果没有任何图像被加载，记录警告但继续执行
        if not imgs_dict['images']:
            self.logger.warning("No images were loaded. Check image paths and availability.")

        # On waymo dataset, the camera coordinate is not the same
        # with common defination, so we need to swap the axes around
        axes_tf = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]])
        top_lidar_ex = np.array([
            [-0.847772463, -0.530354157, -0.002513657, 1.43],
            [0.530355440, -0.847775367, 0.0001801442, 0.0],
            [-0.002226556, -0.001180410, 0.9999968245, 2.184],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # 获取相机参数（使用第一个有效的信息）
        # 修复：使用第一个idx而不是整个idx_list
        if len(idx_list) > 0:
            img_infos = self.infos[idx_list[0]]['image']
            # get the camera related parameters
            for j in range(5):
                cam_name = 'camera_%s' % str(j)
                if f'image_{j}_extrinsic' in img_infos:
                    new_ex_param = np.matmul(axes_tf, np.linalg.inv(img_infos[f'image_{j}_extrinsic']))
                    imgs_dict['extrinsic'][cam_name] = new_ex_param
                    imgs_dict['intrinsic'][cam_name] = img_infos[f'image_{j}_intrinsic']
                    imgs_dict['image_shape'][cam_name] = img_infos.get(f'image_shape_{j}', (0, 0))
        
        return imgs_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from .waymo_eval_detection import WaymoDetectionMetricsEstimator
            eval = WaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False),
                fov_flag=self.dataset_cfg.get('EVAL_FOV_FLAG', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)

        return ap_result_str, ap_dict


if __name__ == '__main__':
    import argparse
    import yaml
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/cpfs2/shared/public/repos/detection/tools/cfgs/det_dataset_cfgs/waymo_one_sweep.yaml', help='specify the config of dataset')
    args = parser.parse_args()

    dataset_cfg = EasyDict(yaml.load(open(args.cfg_file), Loader=yaml.FullLoader))
    class_names=['Vehicle', 'Pedestrian', 'Cyclist']

    file_path ='/cpfs2/shared/public/'
    file_name = 'segment-10868756386479184868_3000_000_3020_000_with_camera_labels'  # training
    # file_name = 'segment-10534368980139017457_4480_000_4500_000_with_camera_labels'  # testing
    info_path = os.path.join(file_path, file_name, '%s.pkl'%file_name)
    with open(info_path, 'rb') as f:
        data_infos = pickle.load(f)


    point_list = []
    point_name = os.listdir(os.path.join(file_path, file_name))
    for name in point_name:
        if len(name) != 8: continue
        point_path = os.path.join(file_path, file_name, name)
        points = np.load(point_path)
        point_list.append(points)

    dataset = WaymoInferenceDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, data_infos=data_infos, point_list=point_list, training=False,
        logger=common_utils.create_logger()
    )
    print("Waymo Inference Dataset contains %d frames." % len(dataset))
    print(dataset[0].keys())
    import pdb; pdb.set_trace()


