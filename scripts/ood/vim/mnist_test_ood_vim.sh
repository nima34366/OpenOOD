#!/bin/bash
# sh scripts/ood/vim/cifar10_test_ood_vim.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# #srun -p mediasuper -x SZ-IDC1-10-112-2-17 --gres=gpu:${GPU} \
# #--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# #--kill-on-bad-exit=1 --job-name=${jobname} \

python main.py \
--config configs/datasets/mnist/mnist.yml \
configs/datasets/mnist/mnist_ood.yml \
configs/networks/lenet.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/vim.yml \
--num_workers 8 \
--network.checkpoint 'results/checkpoints/mnist_lenet_acc99.60.ckpt' \
--mark 0 \
--postprocessor.postprocessor_args.dim 42
