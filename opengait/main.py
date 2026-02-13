"""OpenGait entrypoint with CLI overrides.

This script loads a yaml config (default: ./configs/default.yaml) and lets you override
common dataset-related fields from the command line.
"""

import os
import argparse
import ast

import torch
import torch.nn as nn

from modeling import models
from utils import (
    config_loader,
    get_ddp_module,
    init_seeds,
    params_count,
    get_msg_mgr,
    trainable_params_count,
)

parser = argparse.ArgumentParser(description='Main program for opengait.')

# DDP args (kept for compatibility; not used if you run single-GPU / CPU)
parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
parser.add_argument('--local-rank', type=int, default=0,
                    help="passed by torch.distributed.launch module, for pytorch >=2.0")

# Config + phase
parser.add_argument('--cfgs', type=str, default='./configs/default.yaml',
                    help="path of config file")
parser.add_argument('--phase', default='train', choices=['train', 'test'],
                    help="choose train or test phase")

# Logging / restore
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', default=0, help="iter to restore")
parser.add_argument('--save_name', type=str, default=None,
                    help="Specify the save name for the model")

# Dataset overrides
parser.add_argument('--dataset_root', type=str, default=None,
                    help="Path to the dataset root directory (standard, single-root datasets)")
parser.add_argument(
    '--dataset_type',
    type=str,
    default=None,
    choices=["MultiModal", "MEVID", "CleanGallery", "PerturbedGallery", "Augmented"],
    help="Optional dataset parser type override. If omitted, uses the yaml config.",
)

# MultiModal-specific roots (preferred over stuffing dicts into --dataset_root)
parser.add_argument('--sil_root', type=str, default=None,
                    help="Silhouette root for MultiModal datasets")
parser.add_argument('--rgb_root', type=str, default=None,
                    help="RGB root for MultiModal datasets")

# Clean gallery evaluation
parser.add_argument('--clean_gallery_root', type=str, default=None,
                    help="Clean gallery root (required for CleanGallery evaluation)")

# Perturbed gallery evaluation
parser.add_argument('--perturbed_gallery_root', type=str, default=None,
                    help="Perturbed gallery root (expects <perturb>/<sev>/<pid>/<typ>/<view>/...)")
parser.add_argument('--noise_map_file', type=str, default=None,
                    help="Path to noise_severity_assignments.json (required for PerturbedGallery)")

# Augmented dataset sampling
parser.add_argument('--augmented_root', type=str, default=None,
                    help="Root for the augmented/perturbed training dataset (optional)")
parser.add_argument('--aug_ratio', type=float, default=None,
                    help="Probability of sampling ORIGINAL data (clean ratio). 1-aug_ratio is noisy ratio.")

# Misc data options
parser.add_argument('--data_in_use', type=str, default=None,
                    help="Data-in-use mask list, e.g. '[True, True, False, ...]'")
parser.add_argument('--dataset_partition', type=str, default=None,
                    help="Path to dataset partition json containing TRAIN_SET / TEST_SET")

opt = parser.parse_args()


def initialization(cfgs, training: bool) -> None:
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']

    output_path = os.path.join(
        'output/',
        cfgs['data_cfg']['dataset_name'],
        cfgs['model_cfg']['model'],
        engine_cfg['save_name'],
    )

    if training:
        msg_mgr.init_manager(
            output_path,
            opt.log_to_file,
            engine_cfg['log_iter'],
            engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], int) else 0,
        )
    else:
        msg_mgr.init_logger(output_path, opt.log_to_file)

    msg_mgr.log_info(engine_cfg)

    seed = torch.distributed.get_rank()
    init_seeds(seed)


def run_model(cfgs, training: bool) -> None:
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
        raise ValueError(
            "Expect number of available GPUs({}) equals to the world size({}).".format(
                torch.cuda.device_count(), torch.distributed.get_world_size()
            )
        )

    cfgs = config_loader(opt.cfgs)

    if opt.iter != 0:
        cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
        cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)

    # ----------------------------
    # Apply CLI overrides to cfgs
    # ----------------------------

    if opt.dataset_type is not None:
        cfgs['data_cfg']['dataset_type'] = opt.dataset_type

    # Standard single-root datasets
    if opt.dataset_root is not None:
        cfgs['data_cfg']['dataset_root'] = opt.dataset_root

    # MultiModal datasets: prefer explicit roots
    if opt.sil_root is not None or opt.rgb_root is not None:
        if cfgs['data_cfg'].get('dataset_type') != 'MultiModal':
            # Allow passing sil/rgb roots even if dataset_type is set in yaml.
            cfgs['data_cfg']['dataset_type'] = 'MultiModal'
        base_roots = cfgs['data_cfg'].get('dataset_root')
        if not isinstance(base_roots, dict):
            base_roots = {}
        cfgs['data_cfg']['dataset_root'] = {
            'sil_root': opt.sil_root or base_roots.get('sil_root'),
            'rgb_root': opt.rgb_root or base_roots.get('rgb_root'),
        }

    # Clean gallery
    if opt.clean_gallery_root is not None:
        cfgs['data_cfg']['clean_gallery_root'] = opt.clean_gallery_root

    # Perturbed gallery
    if opt.perturbed_dataset_root is not None:
        cfgs['data_cfg']['perturbed_dataset_root'] = opt.perturbed_dataset_root
    if opt.noise_map_file is not None:
        cfgs['data_cfg']['noise_map_file'] = opt.noise_map_file

    # Augmented training root
    if opt.augmented_root is not None:
        cfgs['data_cfg']['augmented_root'] = opt.augmented_root

    # Aug ratio (default to 1.0 if missing everywhere)
    if opt.aug_ratio is not None:
        cfgs['data_cfg']['aug_ratio'] = opt.aug_ratio
    else:
        cfgs['data_cfg'].setdefault('aug_ratio', 1.0)

    if opt.data_in_use is not None:
        cfgs['data_cfg']['data_in_use'] = ast.literal_eval(opt.data_in_use)

    if opt.dataset_partition is not None:
        cfgs['data_cfg']['dataset_partition'] = opt.dataset_partition

    if opt.save_name is not None:
        cfgs['evaluator_cfg']['save_name'] = opt.save_name
        cfgs['trainer_cfg']['save_name'] = opt.save_name

    training = (opt.phase == 'train')
    initialization(cfgs, training)
    run_model(cfgs, training)
