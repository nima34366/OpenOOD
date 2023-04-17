#!/bin/bash
# sh scripts/basics/mnist/train_mnist.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

# PYTHONPATH='.':$PYTHONPATH \
# #srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# #--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# #--kill-on-bad-exit=1 --job-name=${jobname} \
python3 main.py \
--config configs/datasets/mnist/mnist.yml \
configs/networks/lenet.yml \
configs/pipelines/create_adv.yml \
--network.checkpoint 'results/checkpoints/mnist_lenet_acc99.60.ckpt' \
--network.pretrained True