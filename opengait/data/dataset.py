import os
import pickle
import os.path as osp
import torch.utils.data as tordata
import json
from utils import get_msg_mgr
from tqdm import tqdm
import random


import numpy as np

# to read videos
import decord
import io
import cv2

class DataSet(tordata.Dataset):
    def __init__(self, data_cfg, training):
        """
            seqs_info: the list with each element indicating 
                            a certain gait sequence presented as [label, type, view, paths];
        """
        
        self.multimodal = ("dataset_type" in data_cfg and data_cfg["dataset_type"] == "MultiModal") 
        self.MEVID = ("dataset_type" in data_cfg and data_cfg["dataset_type"] == "MEVID") 
        self.MEVID_multimodal = ("dataset_type" in data_cfg and data_cfg["dataset_type"] == "MEVIDMultiModal") 
        self.cleanGallery = ("dataset_type" in data_cfg and data_cfg["dataset_type"] == "CleanGallery") 
        self.perturbedGallery = ("dataset_type" in data_cfg and data_cfg["dataset_type"] == "PerturbedGallery") 
        self.augmented = ("dataset_type" in data_cfg and data_cfg["dataset_type"] == "Augmented") 

        self.aug_ratio = data_cfg["aug_ratio"]
        
        if self.multimodal and data_cfg["dataset_name"] == "MEVID":
            self.__mevid_multimodal_dataset_parser(data_cfg, training)
        elif self.multimodal:
            self.__multimodal_dataset_parser(data_cfg, training)
        elif self.MEVID:
            self.__mevid_dataset_parser(data_cfg, training)
        elif self.cleanGallery:
            self.__clean_gallery_dataset_parser(data_cfg, training)
        elif self.perturbedGallery:
            self.__perturbed_gallery_dataset_parser(data_cfg, training)
        elif self.augmented:
            self.__augmented_dataset_parser(data_cfg, training)
        else:
            self.__dataset_parser(data_cfg, training)
        self.cache = data_cfg['cache']
        
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list = [seq_info[2] for seq_info in self.seqs_info]

        self.label_set = sorted(list(set(self.label_list)))
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))
        self.seqs_data = [None] * len(self)
        self.indices_dict = {label: [] for label in self.label_set}
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)
        if self.cache:
            self.__load_all_data()

    def __len__(self):
        return len(self.seqs_info)

    def __multimodal_loader__(self, mm_paths):
        
        sil_paths, rgb_paths = mm_paths
        
        if not isinstance(sil_paths, list):
            sil_paths = [sil_paths]
        #print(sil_paths)
        paths = sorted(sil_paths)
        data_list = []

        for pth in paths:
            if pth.endswith('.pkl'):
                # Handle .pkl file loading
                with open(pth, 'rb') as f:
                    _ = pickle.load(f)
                f.close()
                data_list.append(_)  # Add to data_list
                
            elif pth.endswith('.mp4'):
                # Handle .mp4 file processing
                cap = cv2.VideoCapture(pth)

                if not cap.isOpened():
                    print(f"Error: Couldn't open the video {pth}")
                    continue
                
                frames = []  
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break  
                    
                    frame = cv2.resize(frame, (128, 256))
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(gray_frame)  

                cap.release() 
                data_list.append(frames)  
            
            elif pth.endswith('.avi'):
                sil_vid = decord.VideoReader(pth)
                frames = [frm.asnumpy() for frm in sil_vid]

                if len(frames) == 0:
                    print(f"Error: No frames found in {pth}")
                    return  # Skip this file

                # Convert to grayscale & resize
                processed_frames = [
                    cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    for frame in frames
                ]
                data_list.append(np.array(processed_frames))
            else:
                raise ValueError('- Loader - just support .pkl, .avi and .mp4 !!!')

        for idx, data in enumerate(data_list):
            if len(data) != len(data_list[0]):
                raise ValueError(
                    'Each input data({}) should have the same length.'.format(paths[idx]))
            if len(data) == 0:
                raise ValueError(
                    'Each input data({}) should have at least one element.'.format(paths[idx]))

        rgb_paths_ = sorted([rgb_paths])
        #print(rgb_paths_)
        for pth in rgb_paths_:
            if pth.endswith('.pkl'):
                #print("hello")
                # Handle .pkl file loading
                with open(pth, 'rb') as f:
                    _ = pickle.load(f)
                f.close()
                data_list.append(_)  # Add to data_list        
        # try:
        #     print(rgb_paths)
        #     rgb_vid = decord.VideoReader(rgb_paths)
        #     rgb_data = np.array([frm.asnumpy() for frm in rgb_vid])
        #     data_list.append(rgb_data)
        # except Exception as e:
        #     print(f"Error loading RGB video for {rgb_paths}: {str(e)}")
        #     return None
        # print("\n--- Final Data List Characteristics ---")
        # for i, item in enumerate(data_list):
        #     print(f"[{i}] Type: {type(item)}, Length: {len(item) if hasattr(item, '__len__') else 'N/A'}")
        #     if isinstance(item, (list, np.ndarray)) and len(item) > 0 and hasattr(item[0], 'shape'):
        #         print(f"     First element shape: {item[0].shape}")
        return data_list
    
    def __loader__(self, paths):
        paths = sorted(paths)
        data_list = []
        
        for pth in paths:
            if pth.endswith('.pkl'):
                # Handle .pkl file loading
                with open(pth, 'rb') as f:
                    data = pickle.load(f)
                    # if len(data.shape) > 3:
                    #     data = data[..., 0]
                f.close()
                data_list.append(data)  # Add to data_list

            elif pth.endswith('.mp4'):
                # Handle .mp4 file processing
                cap = cv2.VideoCapture(pth)

                if not cap.isOpened():
                    print(f"Error: Couldn't open the video {pth}")
                    continue
                
                frames = []  
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break  
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(gray_frame)  

                cap.release() 
                data_list.append(frames) 

            elif pth.endswith('.avi'):
                
                sil_vid = decord.VideoReader(pth)
                frames = [frm.asnumpy() for frm in sil_vid]
        
                if len(frames) == 0:
                    print(f"Error: No frames found in {pth}")
                    return  
                
                # Convert to grayscale & resize
                processed_frames = [
                    cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (64, 64))
                    for frame in frames
                ]
                processed_frames = np.array(processed_frames)
                data_list.append(processed_frames)
            else:
                raise ValueError('- Loader - just support .pkl, .mp4 and .avi !!!')
            
        for idx, data in enumerate(data_list):
            if len(data) != len(data_list[0]):
                raise ValueError(
                    'Each input data({}) should have the same length.'.format(paths[idx]))
            if len(data) == 0:
                raise ValueError(
                    'Each input data({}) should have at least one element.'.format(paths[idx]))
        return data_list

    def __getitem__(self, idx):
        if not self.cache:
            if self.multimodal:
                data_list = self.__multimodal_loader__(self.seqs_info[idx][-1])
            else:
                data_list = self.__loader__(self.seqs_info[idx][-1])
        elif self.seqs_data[idx] is None:
            if self.multimodal:
                data_list = self.__multimodal_loader__(self.seqs_info[idx][-1])
            else:
                data_list = self.__loader__(self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]
        
        if data_list is None:
            print(f"Skipping corrupted sequence at index {idx}. Trying next index.")
            return self.__getitem__((idx + 1) % len(self))
        
        seq_info = self.seqs_info[idx]
        return data_list, seq_info
    
    def load_pkl(self, pkl_file):
        with open(pkl_file, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def __load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __dataset_parser(self, data_config, training):
        dataset_root = data_config['dataset_root']
        print("Inside data loader")
        try:
            data_in_use = data_config['data_in_use']  # [n], true or false
        except:
            data_in_use = None

        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (
            train_set + test_set)]
        msg_mgr = get_msg_mgr()

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug('-------- Miss Pid List --------')
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            skipped_files = []
            for lab in label_set:
                for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                        seq_info = [lab, typ, vie]
                        seq_path = osp.join(dataset_root, *seq_info)
                        seq_dirs = sorted(os.listdir(seq_path))
                        if seq_dirs != []:
                            seq_dirs = [osp.join(seq_path, dir)
                                        for dir in seq_dirs]
                            if data_in_use is not None:
                                seq_dirs = [dir for dir, use_bl in zip(
                                    seq_dirs, data_in_use) if use_bl]
                            seqs_info_list.append([*seq_info, seq_dirs])
                            # seq_dirs_new = []
                            # for item in seq_dirs:
                            #     if os.path.getsize(item) == 0:
                            #         print(f"Error: File is empty - {item}")
                            #         continue
                            #     else:
                            #         item_pkl = self.load_pkl(item)
                            #         if data_config['dataset_name'] in ['CCPG','SUSTech1K'] and len(item_pkl.shape) != 3:
                            #             print(f"skipping file {item}")
                            #             skipped_files.append([*seq_info, seq_dirs, len(item_pkl.shape)]) 
                            #             continue
                            #         else:
                            #             seq_dirs_new.append(item)
                            # if seq_dirs_new != []:
                            #     seqs_info_list.append([*seq_info, seq_dirs_new])
                            # else:
                            #     msg_mgr.log_debug(
                            #     '.pkl files empty in %s-%s-%s.' % (lab, typ, vie))
                            #print(seqs_info_list)
                        else:
                            msg_mgr.log_debug(
                                'Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
            return seqs_info_list

        self.seqs_info = get_seqs_info_list(
            train_set) if training else get_seqs_info_list(test_set)
        
    def __multimodal_dataset_parser(self, data_config, training):
        sil_dataset_root = data_config['dataset_root']["sil_root"]
        img_dataset_root = data_config['dataset_root']["rgb_root"]
        
        try:
            data_in_use = data_config['data_in_use']  # [n], true or false
        except:
            data_in_use = None

        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(sil_dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (
            train_set + test_set)]
        msg_mgr = get_msg_mgr()

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug('-------- Miss Pid List --------')
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            for lab in label_set:
                for typ in sorted(os.listdir(osp.join(sil_dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(sil_dataset_root, lab, typ))):
                        seq_info = [lab, typ, vie]
                        seq_path = osp.join(sil_dataset_root, *seq_info)
                        seq_dirs = sorted(os.listdir(seq_path))
                        if seq_dirs != []:
                            seq_dirs = [osp.join(seq_path, dir)
                                        for dir in seq_dirs]
                            if data_in_use is not None:
                                seq_dirs = [dir for dir, use_bl in zip(
                                    seq_dirs, data_in_use) if use_bl]
                                
                            # adding support for multiple modalities
                            
                            #img_path = osp.join(img_dataset_root, lab,"-".join(seq_info)+".avi")
                            # if osp.exists(img_path):
                            #     seq_dirs.append(img_path)
                            #     seqs_info_list.append([*seq_info, seq_dirs])

                            img_path = osp.join(img_dataset_root, *seq_info)
                            # print(img_path)
                            img_path_dirs = sorted(os.listdir(img_path))
                            # for dir in img_path_dirs:
                            #     print(osp.join(img_path, dir))
                            if img_path_dirs != []:
                                seq_dirs.extend(osp.join(img_path, dir) for dir in img_path_dirs)
                                seqs_info_list.append([*seq_info, seq_dirs])

                            else: # if the .avi file doesnt exist show logs
                                print("Path not found")
                                msg_mgr.log_debug(
                                    f"Found no rgb .avi file at {img_path}"
                                )
                            
                        else:
                            msg_mgr.log_debug(
                                'Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
            #print(seqs_info_list)
            return seqs_info_list

        self.seqs_info = get_seqs_info_list(
            train_set) if training else get_seqs_info_list(test_set)

    def __mevid_dataset_parser(self, data_config, training):

        ##-----------MEVID train and test sets are stored seperately-----------
        if training:
            dataset_root = data_config['dataset_root']
        else:
            dataset_root = data_config['dataset_root_test']
        ##---------------------------------------------------------------------

        try:
            data_in_use = data_config['data_in_use']  # [n], true or false
        except:
            data_in_use = None

        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (
            train_set + test_set)]
        msg_mgr = get_msg_mgr()

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug('-------- Miss Pid List --------')
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)

        ### ---------------------------- updated to handle MEVID dataset structure --------------------------------
        def get_seqs_info_list(label_set):
            seqs_info_list = []
            for lab in label_set:
                for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                        seq_info = [lab, typ, vie]
                        seq_path = osp.join(dataset_root, *seq_info)
                        seq_dirs = sorted(os.listdir(seq_path))  # m
                        if seq_dirs != []:
                            for seq_dir in seq_dirs:
                                tracklet_num = seq_dir.split(".")[0]
                                s_info = [lab,typ,vie + "_" + str(tracklet_num)]
                                seq_dir = [osp.join(seq_path, seq_dir)]

                                ### Change here to s_info to include tracklet information
                                seqs_info_list.append([*seq_info, seq_dir])
                        else:
                            msg_mgr.log_debug(
                                'Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
            return seqs_info_list
        ### ----------------------------------------------------------------------------------------------------------

        self.seqs_info = get_seqs_info_list(
            train_set) if training else get_seqs_info_list(test_set)
    
    def __mevid_multimodal_dataset_parser(self, data_config, training):

        ##-----------MEVID train and test sets are stored seperately-----------
        if training:
            sil_dataset_root = data_config['dataset_root']["sil_root"]
            img_dataset_root = data_config['dataset_root']["rgb_root"]
        else:
            sil_dataset_root = data_config['dataset_root_test']["sil_root"]
            img_dataset_root = data_config['dataset_root_test']["rgb_root"]
        ##---------------------------------------------------------------------

        try:
            data_in_use = data_config['data_in_use']  # [n], true or false
        except:
            data_in_use = None

        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(sil_dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (
            train_set + test_set)]
        msg_mgr = get_msg_mgr()

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug('-------- Miss Pid List --------')
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)

        ### ---------------------------- updated to handle MEVID dataset structure --------------------------------
        def get_seqs_info_list(label_set):
            seqs_info_list = []
            for lab in label_set:
                for typ in sorted(os.listdir(osp.join(sil_dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(sil_dataset_root, lab, typ))):
                        seq_info = [lab, typ, vie]
                        seq_path = osp.join(sil_dataset_root, *seq_info)
                        seq_dirs = sorted(os.listdir(seq_path))  # m
                        if seq_dirs != []:
                            for seq_dir in seq_dirs:
                                tracklet_num = seq_dir.split(".")[0]
                                s_info = [lab,typ,vie + "_" + str(tracklet_num)]
                                seq_info.append(tracklet_num)
                                seq_dir = [osp.join(seq_path, seq_dir)]
                                
                                # adding support for multiple modalities
                                img_path = osp.join(img_dataset_root, lab,"-".join(seq_info)+".avi")
                                if osp.exists(img_path):
                                    seq_dir.append(img_path)
                                    seqs_info_list.append([*s_info, seq_dir])
                                    
                                else: # if the .avi file doesnt exist show logs
                                    msg_mgr.log_debug(
                                        f"Found no rgb .avi file at {img_path}"
                                )
                        else:
                            msg_mgr.log_debug(
                                'Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
            return seqs_info_list
        ### ----------------------------------------------------------------------------------------------------------

        self.seqs_info = get_seqs_info_list(
            train_set) if training else get_seqs_info_list(test_set)
        
    

    def __clean_gallery_dataset_parser(self, data_config, training):
        dataset_root = data_config['dataset_root']
        clean_gallery = True
        
        clean_gallery_root = "/home/c3-0/datasets/robust_gait/casiab/sil_pkl/orig/schp"                   

        if data_config['dataset_name'] == 'CASIA-B':
            gallery_typs = ['nm-01', 'nm-02', 'nm-03', 'nm-04']
                
        try:
            data_in_use = data_config['data_in_use']  # [n], true or false
        except:
            data_in_use = None

        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (
            train_set + test_set)]
        msg_mgr = get_msg_mgr()

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug('-------- Miss Pid List --------')
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)
        
        def get_seqs_info_list(label_set):
            flag = 0
            seqs_info_list = []
            skipped_files1 = []
            not_skipped_files1 = []
            skipped_files = []
            for lab in tqdm(label_set, desc='get_seqs_info_list'):
                for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                        seq_info = [lab, typ, vie]

                        if (clean_gallery) and (not training) and (typ in gallery_typs):
                            seq_path = osp.join(clean_gallery_root, *seq_info)
                        else:
                            seq_path = osp.join(dataset_root, *seq_info)
                        
                        seq_dirs = sorted(os.listdir(seq_path))

                        if seq_dirs != []:
                            seq_dirs = [osp.join(seq_path, dir)
                                        for dir in seq_dirs]
                            if data_in_use is not None:
                                seq_dirs = [dir for dir, use_bl in zip(seq_dirs, data_in_use) if use_bl]

                            seq_dirs_new = []
                            for item in seq_dirs:
                                # if len(self.load_pkl(item)) == 0:
                                item_pkl = self.load_pkl(item)
                                if data_config['dataset_name'] in ['CCPG','SUSTech1K'] and len(item_pkl.shape) < 3:
                                    # print(item)
                                    skipped_files.append([*seq_info, seq_dirs, len(item_pkl.shape)]) 
                                    continue
                                else:
                                    # if len(item_pkl.shape) == 4:

                                    seq_dirs_new.append(item)
                                # import pdb; pdb.set_trace()
                            if seq_dirs_new != []:
                                seqs_info_list.append([*seq_info, seq_dirs_new])
                            else:
                                msg_mgr.log_debug(
                                '.pkl files empty in %s-%s-%s.' % (lab, typ, vie))
                        else:
                            # print('Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
                            msg_mgr.log_debug(
                                'Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
            return seqs_info_list
                            
        self.seqs_info = get_seqs_info_list(
            train_set) if training else get_seqs_info_list(test_set)
        
    def __perturbed_gallery_dataset_parser(self, data_config, training):
        with open(f"/home/re207167/OpenGait/datasets/{data_config['dataset_name']}/noise_severity_assignments.json", "r") as file:
            noise_sev_map = json.load(file)

        dataset_root = data_config['dataset_root']
        perturbed_gallery = True                   

        if data_config['dataset_name'] == 'CASIA-B':
            gallery_typs = ['nm-01', 'nm-02', 'nm-03', 'nm-04']
        elif data_config['dataset_name'] == 'SUSTech1K':
            gallery_typs = ['00-nm']
                
        try:
            data_in_use = data_config['data_in_use']  # [n], true or false
        except:
            data_in_use = None

        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (
            train_set + test_set)]
        msg_mgr = get_msg_mgr()

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug('-------- Miss Pid List --------')
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)
        
        def get_seqs_info_list(label_set):
            flag = 0
            seqs_info_list = []
            skipped_files1 = []
            not_skipped_files1 = []
            skipped_files = []
            for lab in tqdm(label_set, desc='get_seqs_info_list'):
                for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                        
                        seq_info = [lab, typ, vie]

                        if (perturbed_gallery) and (not training) and (typ in gallery_typs):
                            video_name = f"{lab}-{typ}-{vie}"
                            perturb, sev = noise_sev_map[video_name]
                            if data_config['dataset_name'] == 'CASIA-B':
                                perturbed_gallery_root = f"/home/c3-0/datasets/robust_gait/casiab/sil_pkl/perturb/{perturb}/{sev}"
                            elif data_config['dataset_name'] == 'SUSTech1K':
                                perturbed_gallery_root = f"/home/c3-0/datasets/robust_gait/sustech/sil_pkl/perturb/{perturb}/{sev}"
                            
                            seq_path = osp.join(perturbed_gallery_root, *seq_info)
                        else:
                            seq_path = osp.join(dataset_root, *seq_info)
                        
                        seq_dirs = sorted(os.listdir(seq_path))

                        if seq_dirs != []:
                            seq_dirs = [osp.join(seq_path, dir)
                                        for dir in seq_dirs]
                            if data_in_use is not None:
                                seq_dirs = [dir for dir, use_bl in zip(seq_dirs, data_in_use) if use_bl]
                            
                            seqs_info_list.append([*seq_info, seq_dirs])

                            # seq_dirs_new = []
                            # for item in seq_dirs:
                            #     # if len(self.load_pkl(item)) == 0:
                            #     if os.path.getsize(item) == 0:
                            #         print(f"Error: File is empty - {item}")
                            #         continue
                            #     else:
                            #         item_pkl = self.load_pkl(item)
                            #         if data_config['dataset_name'] in ['CCPG','SUSTech1K'] and len(item_pkl.shape) != 3:
                            #             # print(item)
                            #             print(f"skipping file {item}")
                            #             skipped_files.append([*seq_info, seq_dirs, len(item_pkl.shape)]) 
                            #             continue
                            #         else:
                            #             seq_dirs_new.append(item)

                            # if seq_dirs_new != []:
                            #     seqs_info_list.append([*seq_info, seq_dirs_new])
                            # else:
                            #     msg_mgr.log_debug(
                            #     '.pkl files empty in %s-%s-%s.' % (lab, typ, vie))
                        else:
                            # print('Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
                            msg_mgr.log_debug(
                                'Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
            return seqs_info_list
                  
        self.seqs_info = get_seqs_info_list(
            train_set) if training else get_seqs_info_list(test_set)

    def __augmented_dataset_parser(self, data_config, training):
        
        dataset_root = data_config['dataset_root']
        dataset_map = {'CASIA-B':'casiab','SUSTech1K':'sustech','CCPG':'ccpg'}
        perturbed_dataset_root = f'/home/re207167/{dataset_map[data_config["dataset_name"]]}_schp_perturbed_sil_pkl_dataset'
        try:
            data_in_use = data_config['data_in_use']  # [n], true or false
        except:
            data_in_use = None

        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (
            train_set + test_set)]
        msg_mgr = get_msg_mgr()

        clean_ratio = float(self.aug_ratio)
        noisy_ratio = float(1 - self.aug_ratio)

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug('-------- Miss Pid List --------')
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            skipped_files = []
            for lab in label_set:
                for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                        seq_info = [lab, typ, vie]

                        dataset_choice = random.choices(['original', 'augmented'], weights=[clean_ratio, noisy_ratio], k=1)[0]
                        if dataset_choice == 'augmented':
                            augmented_path = osp.join(perturbed_dataset_root, *seq_info)
                            if osp.exists(augmented_path): 
                                seq_path = augmented_path
                            else:
                                seq_path = osp.join(dataset_root, *seq_info)
                        else:
                            seq_path = osp.join(dataset_root, *seq_info)

                        seq_dirs = sorted(os.listdir(seq_path))
                        if seq_dirs != []:
                            seq_dirs = [osp.join(seq_path, dir)
                                        for dir in seq_dirs]
                            if data_in_use is not None:
                                seq_dirs = [dir for dir, use_bl in zip(
                                    seq_dirs, data_in_use) if use_bl]
                                
                            seq_dirs_new = []
                            for item in seq_dirs:
                                if os.path.getsize(item) == 0:
                                    print(f"Error: File is empty - {item}")
                                    continue
                                else:
                                    item_pkl = self.load_pkl(item)
                                    if data_config['dataset_name'] in ['CCPG','SUSTech1K'] and len(item_pkl.shape) != 3:
                                        # print(item)
                                        #
                                        # (f"skipping file {item}")
                                        skipped_files.append([*seq_info, seq_dirs, len(item_pkl.shape)]) 
                                        continue
                                    else:
                                        seq_dirs_new.append(item)
                            if seq_dirs_new != []:
                                seqs_info_list.append([*seq_info, seq_dirs_new])
                            else:
                                msg_mgr.log_debug(
                                '.pkl files empty in %s-%s-%s.' % (lab, typ, vie))
                        else:
                            msg_mgr.log_debug(
                                'Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
            return seqs_info_list

        self.seqs_info = get_seqs_info_list(
            train_set) if training else get_seqs_info_list(test_set)
    
    def __casiab_augmented_dataset_parser_2(self, data_config, training):
        
        dataset_root = data_config['dataset_root']
        perturbed_dataset_root = '/home/re207167/casiab_schp_perturbed_sil_pkl_dataset'
        try:
            data_in_use = data_config['data_in_use']  # [n], true or false
        except:
            data_in_use = None

        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (
            train_set + test_set)]
        msg_mgr = get_msg_mgr()

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug('-------- Miss Pid List --------')
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            for lab in label_set:
                for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                        seq_info = [lab, typ, vie]

                        seq_path = osp.join(dataset_root, *seq_info)
                        seq_dirs = sorted(os.listdir(seq_path))
                        if seq_dirs != []:
                            seq_dirs = [osp.join(seq_path, dir)
                                        for dir in seq_dirs]
                            # per_seq_dirs = sorted(os.listdir(seq_path))
                            # per_seq_path = osp.join(perturbed_dataset_root, *seq_info)
                            # for dir in per_seq_dirs:
                            #     seq_dirs.append(osp.join(per_seq_path, dir))

                            if data_in_use is not None:
                                seq_dirs = [dir for dir, use_bl in zip(
                                    seq_dirs, data_in_use) if use_bl]
                            seqs_info_list.append([*seq_info, seq_dirs])
                        else:
                            msg_mgr.log_debug(
                                'Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
                        
                        per_seq_path = osp.join(perturbed_dataset_root, *seq_info)
                        per_seq_dirs = sorted(os.listdir(per_seq_path))
                        if per_seq_dirs != []:
                            per_seq_dirs = [osp.join(per_seq_path, dir)
                                        for dir in per_seq_dirs]
                            seqs_info_list.append([*seq_info, per_seq_dirs])
                        else:
                            msg_mgr.log_debug(
                                'Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
            
            return seqs_info_list

        self.seqs_info = get_seqs_info_list(
            train_set) if training else get_seqs_info_list(test_set)