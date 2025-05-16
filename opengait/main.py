
import os
import argparse
import torch
import torch.nn as nn
from modeling import models
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr, trainable_params_count

parser = argparse.ArgumentParser(description='Main program for opengait.')
parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
parser.add_argument('--local-rank', type=int, default=0,
                    help="passed by torch.distributed.launch module, for pytorch >=2.0")
parser.add_argument('--cfgs', type=str,
                    default='config/default.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', default=0, help="iter to restore")
parser.add_argument('--save_name', type=str, default=None, help="Specify the save name for the model")
parser.add_argument('--dataset_root', type=str, default=None, help="Path to the dataset root directory")
parser.add_argument('--dataset_type', type=str, default=None, help="Type of the dataset parser")
parser.add_argument('--aug_ratio', type=float, default=None, help="Percentage of noisy data to be added to the train set")
opt = parser.parse_args()

# Function to calculate and print total parameters of each main component
def print_total_params(model):
    # Get the total parameters in sil_encoder
    sil_encoder_params = sum(p.numel() for p in model.sil_encoder.parameters())
    print(f"Total parameters in Silhouette Encoder (sil_encoder): {sil_encoder_params}")

    # Get the total parameters in sil_proj (silhouette projection head)
    sil_proj_params = sum(p.numel() for p in model.sil_proj.parameters())
    print(f"Total parameters in Silhouette Projection Head (sil_proj): {sil_proj_params}")

    # Get the total parameters in rgb_encoder
    rgb_encoder_params = sum(p.numel() for p in model.rgb_encoder.parameters())
    print(f"Total parameters in RGB Encoder (rgb_encoder): {rgb_encoder_params}")

    # Get the total parameters in rgb_proj (RGB projection head)
    rgb_proj_params = sum(p.numel() for p in model.rgb_proj.parameters())
    print(f"Total parameters in RGB Projection Head (rgb_proj): {rgb_proj_params}")


def initialization(cfgs, training):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])
    if training:
        msg_mgr.init_manager(output_path, opt.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, opt.log_to_file)

    msg_mgr.log_info(engine_cfg)

    seed = torch.distributed.get_rank()
    init_seeds(seed)


def run_model(cfgs, training):
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model']) 
    model = Model(cfgs, training)

    if training and cfgs['trainer_cfg']['sync_BN']:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cfgs['trainer_cfg']['fix_BN']:
        model.fix_BN()

    model = get_ddp_module(model, cfgs['trainer_cfg']['find_unused_parameters'])
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info(trainable_params_count(model))
    
    msg_mgr.log_info("Model Initialization Finished!")

    if training:
        Model.run_train(model)
    else:
        Model.run_test(model)


if __name__ == '__main__':
    torch.distributed.init_process_group('nccl', init_method='env://')
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of available GPUs({}) equals to the world size({}).".format(
            torch.cuda.device_count(), torch.distributed.get_world_size()))
    cfgs = config_loader(opt.cfgs)
    if opt.iter != 0:
        cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
        cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)

    if opt.dataset_root is not None:
        cfgs["data_cfg"]["dataset_root"] = opt.dataset_root
    
    if opt.dataset_type is not None:
        cfgs["data_cfg"]["dataset_type"] = opt.dataset_type
    
    if opt.aug_ratio is not None:
        cfgs["data_cfg"]["aug_ratio"] = opt.aug_ratio
    else:
        cfgs["data_cfg"]["aug_ratio"] = 1.0

    if opt.save_name is not None:
        cfgs["evaluator_cfg"]["save_name"] = opt.save_name
        cfgs["trainer_cfg"]["save_name"] = opt.save_name

    training = (opt.phase == 'train')
    initialization(cfgs, training)
    run_model(cfgs, training)
