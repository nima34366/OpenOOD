#!/bin/bash
# sh scripts/osr/openmax/cifar100_test_ood_openmax.sh

# GPU=1
# CPU=1
# node=30
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/openmax.yml \
--num_workers 8 \
--network.checkpoint 'results/checkpoints/cifar100_res18_acc78.20.ckpt' \
--mark 0
