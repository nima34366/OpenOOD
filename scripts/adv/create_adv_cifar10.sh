#!/bin/bash
# sh scripts/basics/cifar10/train_cifar10.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

# PYTHONPATH='.':$PYTHONPATH \
# #srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# #--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# #--kill-on-bad-exit=1 --job-name=${jobname} \
python3 main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/create_adv.yml \
--network.checkpoint 'results/checkpoints/cifar10_res18_acc94.30.ckpt' \
--network.pretrained True