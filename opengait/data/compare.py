import math
import random
import numpy as np
from utils import get_msg_mgr


class CollateFn(object):
    def __init__(self, label_set, sample_config):
        self.label_set = label_set
        self.num_modalities = sample_config["modalities"] if "modalities" in sample_config else 1

        
        sample_type = sample_config['sample_type']
        sample_type = sample_type.split('_')
        self.sampler = sample_type[0]
        self.ordered = sample_type[1]
        
        if self.num_modalities > 1:
            
            self.modality_config = None
            
            if "modality" in sample_config:
                self.modality_config = sample_config["modality"]
        
        if self.sampler not in ['fixed', 'unfixed', 'all']:
            raise ValueError
        if self.ordered not in ['ordered', 'unordered']:
            raise ValueError
        self.ordered = sample_type[1] == 'ordered'

        # fixed cases
        if self.sampler == 'fixed':
            self.frames_num_fixed = sample_config['frames_num_fixed']

        # unfixed cases
        if self.sampler == 'unfixed':
            self.frames_num_max = sample_config['frames_num_max']
            self.frames_num_min = sample_config['frames_num_min']

        if self.sampler != 'all' and self.ordered:
            self.frames_skip_num = sample_config['frames_skip_num']

        self.frames_all_limit = -1
        if self.sampler == 'all' and 'frames_all_limit' in sample_config:
            self.frames_all_limit = sample_config['frames_all_limit']

    def __call__(self, batch):
        batch_size = len(batch)
        # currently, the functionality of feature_num is not fully supported yet, it refers to 1 now. We are supposed to make our framework support multiple source of input data, such as silhouette, or skeleton.
        if self.num_modalities>1:
            feature_num = 1
        else:
            feature_num = len(batch[0][0])
        
        seqs_batch, labs_batch, typs_batch, vies_batch = [], [], [], []
        
        if self.num_modalities>1:
            seqs2_batch = []

        for bt in batch:
            if self.num_modalities>1:
                seqs_batch.append([bt[0][0]])
                seqs2_batch.append([bt[0][1]])
            else:
                seqs_batch.append(bt[0])
            labs_batch.append(self.label_set.index(bt[1][0]))
            typs_batch.append(bt[1][1])
            vies_batch.append(bt[1][2])

        global count
        count = 0

        def sample_frames(seqs):
            global count
            sampled_fras = [[] for i in range(feature_num)]
            seq_len = len(seqs[0])
            indices = list(range(seq_len))

            if self.sampler in ['fixed', 'unfixed']:
                if self.sampler == 'fixed':
                    frames_num = self.frames_num_fixed
                else:
                    frames_num = random.choice(
                        list(range(self.frames_num_min, self.frames_num_max+1)))

                if self.ordered:
                    fs_n = frames_num + self.frames_skip_num
                    if seq_len < fs_n:
                        it = math.ceil(fs_n / seq_len)
                        seq_len = seq_len * it
                        indices = indices * it

                    start = random.choice(list(range(0, seq_len - fs_n + 1)))
                    end = start + fs_n
                    idx_lst = list(range(seq_len))
                    idx_lst = idx_lst[start:end]
                    idx_lst = sorted(np.random.choice(
                        idx_lst, frames_num, replace=False))
                    indices = [indices[i] for i in idx_lst]
                else:
                    replace = seq_len < frames_num

                    if seq_len == 0:
                        get_msg_mgr().log_debug('Find no frames in the sequence %s-%s-%s.'
                                                % (str(labs_batch[count]), str(typs_batch[count]), str(vies_batch[count])))

                    count += 1
                    indices = np.random.choice(
                        indices, frames_num, replace=replace)

            for i in range(feature_num):
                for j in indices[:self.frames_all_limit] if self.frames_all_limit > -1 and len(indices) > self.frames_all_limit else indices:
                    sampled_fras[i].append(seqs[i][j])
            return sampled_fras

        # f: feature_num
        # b: batch_size
        # p: batch_size_per_gpu
        # g: gpus_num
        
        fras_batch = [sample_frames(seqs) for seqs in seqs_batch]  # [b, f]
        
        batch = [fras_batch, labs_batch, typs_batch, vies_batch, None]

        if self.sampler == "fixed":
            fras_batch = [[np.asarray(fras_batch[i][j]) for i in range(batch_size)]
                          for j in range(feature_num)]  # [f, b]
        else:
            seqL_batch = [[len(fras_batch[i][0])
                           for i in range(batch_size)]]  # [1, p]

            def my_cat(k): return np.concatenate(
                [fras_batch[i][k] for i in range(batch_size)], 0)
            fras_batch = [[my_cat(k)] for k in range(feature_num)]  # [f, g]

            batch[-1] = np.asarray(seqL_batch)
        
        if self.num_modalities>1:
            
            temp = self.frames_all_limit
            
            if self.modality_config is not None:
            
                self.frames_all_limit = self.modality_config["rgb"]["frames_num_fixed"]
            
            fras2_batch = [sample_frames(seqs) for seqs in seqs2_batch]  # [b, f]
            
            self.frames_all_limit = temp
            
            if self.sampler == "fixed":
                fras2_batch = [[np.asarray(fras2_batch[i][j]) for i in range(batch_size)]
                            for j in range(feature_num)]  # [f, b]
            else:
                seqL_batch = [[len(fras2_batch[i][0])
                            for i in range(batch_size)]]  # [1, p]

                def my_cat(k): return np.concatenate(
                    [fras2_batch[i][k] for i in range(batch_size)], 0)
                fras2_batch = [[my_cat(k)] for k in range(feature_num)]  # [f, g]

                # batch[-1] = np.asarray(seqL_batch)
                
            fras_batch = fras_batch + fras2_batch
            
        batch[0] = fras_batch
        
        return batch






class MultiCollateFn(CollateFn):
    def __init__(self, label_set, sample_config):
        
        super(MultiCollateFn, self).__init__(label_set, sample_config)
        
    def __call__(self, batch):

        batch_size = len(batch)
        #print(f"Batch size: {batch_size}")
        # currently, the functionality of feature_num is not fully supported yet, it refers to 1 now. We are supposed to make our framework support multiple source of input data, such as silhouette, or skeleton.
        if self.num_modalities>1:
            feature_num = 1
        else:
            feature_num = len(batch[0][0])
        
        seqs_batch, labs_batch, typs_batch, vies_batch = [], [], [], []
        
        #print(f"Num modalities:{self.num_modalities}")
        if self.num_modalities>1:
            seqs1_batch = []
            seqs2_batch = []
        #ct = 0
        for bt in batch:
            
            if self.num_modalities>1:
                seqs_batch.append([bt[0][0]])
                seqs1_batch.append([bt[0][1]])
                #seqs2_batch.append([bt[0][2]])
            else:
                seqs_batch.append(bt[0])
            labs_batch.append(self.label_set.index(bt[1][0]))
            typs_batch.append(bt[1][1])
            vies_batch.append(bt[1][2])

            # if ct == 0:
            #     print(type(bt[0][0]))
            #     print(type(bt[0][1]))
            #     #print(bt)
            #     print(len(bt))
            #     print(labs_batch)
            #     print(typs_batch)
            #     print(vies_batch)
            #     ct += 1

        global count
        count = 0

        def sample_frames(seqs):
            global count
            sampled_fras = [[] for i in range(feature_num)]
            seq_len = len(seqs[0])
            indices = list(range(seq_len))

            if self.sampler in ['fixed', 'unfixed']:
                if self.sampler == 'fixed':
                    frames_num = self.frames_num_fixed
                else:
                    frames_num = random.choice(
                        list(range(self.frames_num_min, self.frames_num_max+1)))

                if self.ordered:
                    fs_n = frames_num + self.frames_skip_num
                    if seq_len < fs_n:
                        it = math.ceil(fs_n / seq_len)
                        seq_len = seq_len * it
                        indices = indices * it

                    start = random.choice(list(range(0, seq_len - fs_n + 1)))
                    end = start + fs_n
                    idx_lst = list(range(seq_len))
                    idx_lst = idx_lst[start:end]
                    idx_lst = sorted(np.random.choice(
                        idx_lst, frames_num, replace=False))
                    indices = [indices[i] for i in idx_lst]
                else:
                    replace = seq_len < frames_num

                    if seq_len == 0:
                        get_msg_mgr().log_debug('Find no frames in the sequence %s-%s-%s.'
                                                % (str(labs_batch[count]), str(typs_batch[count]), str(vies_batch[count])))

                    count += 1
                    indices = np.random.choice(
                        indices, frames_num, replace=replace)

            for i in range(feature_num):
                for j in indices[:self.frames_all_limit] if self.frames_all_limit > -1 and len(indices) > self.frames_all_limit else indices:
                    sampled_fras[i].append(seqs[i][j])
            return sampled_fras

        # f: feature_num
        # b: batch_size
        # p: batch_size_per_gpu
        # g: gpus_num
        
        fras_batch = [sample_frames(seqs) for seqs in seqs_batch]  # [b, f]
        
        batch = [fras_batch, labs_batch, typs_batch, vies_batch, None]

        if self.sampler == "fixed":
            fras_batch = [[np.asarray(fras_batch[i][j]) for i in range(batch_size)]
                          for j in range(feature_num)]  # [f, b]
        else:
            seqL_batch = [[len(fras_batch[i][0])
                           for i in range(batch_size)]]  # [1, p]

            def my_cat(k): return np.concatenate(
                [fras_batch[i][k] for i in range(batch_size)], 0)
            fras_batch = [[my_cat(k)] for k in range(feature_num)]  # [f, g]

            batch[-1] = np.asarray(seqL_batch)
        
        if self.num_modalities>1:
            
            # for rgb type data
            
            temp = self.sampler
            
            if self.modality_config is not None:
            
                # self.frames_all_limit = self.modality_config["rgb"]["frames_num_fixed"]
                self.sampler = "fixed"
                self.frames_num_fixed = self.modality_config["rgb"]["frames_num_fixed"]
                self.frames_skip_num = self.modality_config["rgb"]["frames_skip_num"]
            
            fras1_batch = [sample_frames(seqs) for seqs in seqs1_batch]  # [b, f]
            
            
            if self.sampler == "fixed":
                fras1_batch = [[np.asarray(fras1_batch[i][j]) for i in range(batch_size)]
                            for j in range(feature_num)]  # [f, b]
            else:
                seqL_batch = [[len(fras1_batch[i][0])
                            for i in range(batch_size)]]  # [1, p]
            
                def my_cat(k): return np.concatenate(
                    [fras1_batch[i][k] for i in range(batch_size)], 0)
                fras1_batch = [[my_cat(k)] for k in range(feature_num)]  # [f, g]

                # batch[-1] = np.asarray(seqL_batch)
            
            self.sampler = temp
                
            fras_batch = fras_batch + fras1_batch
            
            
            temp = self.frames_all_limit
            
            
        batch[0] = fras_batch
        
        return batch
