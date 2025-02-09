#!/bin/bash
# sh scripts/osr/opengan/cifar10_train_opengan.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
##srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
##--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
##--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/networks/opengan.yml \
configs/pipelines/train/train_opengan.yml \
configs/preprocessors/base_preprocessor.yml \
--num_workers 8 \
--dataset.feat_path 'results/cifar10_resnet18_32x32_feat_extract/feature.npz' \
--network.backbone.pretrained True \
--network.backbone.checkpoint 'results/checkpoints/cifar10_res18_acc94.30.ckpt'
