#!/bin/bash
# sh scripts/ood/react/cifar100_test_ood_react.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
##srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
##--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
##--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/react_net.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/react.yml \
--network.pretrained False \
--network.backbone.name resnet18_32x32 \
--network.backbone.pretrained True \
--network.backbone.checkpoint 'results/checkpoints/cifar100_res18_acc78.20.ckpt' \
--num_workers 8 \
--mark 0
