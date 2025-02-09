#!/bin/bash
# sh scripts/ood/mcd/cifar10_test_mcd.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# #srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# #--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# #--kill-on-bad-exit=1 --job-name=${jobname} \
# -w SG-IDC1-10-51-2-${node} \
python -m pdb -c continue main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/mcd_net.yml \
configs/preprocessors/base_preprocessor.yml \
configs/pipelines/test/test_ood.yml \
configs/postprocessors/mcd.yml \
--network.backbone.name resnet18_32x32 \
--network.pretrained True \
--network.checkpoint 'results/cifar10_oe_mcd_mcd_e100_lr0.1/last_epoch100_acc0.7520.ckpt' \
--num_workers 4
