#!/bin/bash
# sh scripts/basics/imagenet/train_imagenet.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

# PYTHONPATH='.':$PYTHONPATH \
# #srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# #--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# #--kill-on-bad-exit=1 --job-name=${jobname} \
python3 main.py \
--config configs/datasets/imagenet/imagenet.yml \
configs/networks/resnet50.yml \
configs/pipelines/create_adv.yml \
--network.checkpoint 'results/checkpoints/imagenet_res50_acc76.10.pth' \
--network.pretrained True