#!/bin/bash
# sh scripts/ood/kl_matching/cifar100_test_ood_kl_matching.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# #srun -p mediasuper -x SZ-IDC1-10-112-2-17 --gres=gpu:${GPU} \
# #--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# #--kill-on-bad-exit=1 --job-name=${jobname} \

python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/klm.yml \
--num_workers 8 \
--network.checkpoint 'results/checkpoints/cifar100_res18_acc78.20.ckpt' \
--mark 0
