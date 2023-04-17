#!/bin/bash
# sh scripts/ood/godin/imagenet_test_ood_godin.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# #srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# #--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# #--kill-on-bad-exit=1 --job-name=${jobname} \
# -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/imagenet/imagenet.yml \
configs/datasets/imagenet/imagenet_ood.yml \
configs/networks/godin_net.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/godin.yml \
--network.backbone.name resnet50 \
--num_workers 8 \
--network.checkpoint 'results/imagenet_godin_net_base_e100_lr0.1_default/best.ckpt' \
--mark epoch_100
