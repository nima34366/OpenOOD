#!/bin/bash
# sh scripts/ood/vim/imagenet_test_ood_vim.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/imagenet/imagenet.yml \
configs/datasets/imagenet/imagenet_ood.yml \
configs/networks/resnet50.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/vim.yml \
--num_workers 4 \
--ood_dataset.image_size 256 \
--dataset.test.batch_size 256 \
--dataset.val.batch_size 256 \
--network.pretrained True \
--network.checkpoint 'results/checkpoints/imagenet_res50_acc76.10.pth' \
--postprocessor.postprocessor_args.dim 1000 \
--merge_option merge
