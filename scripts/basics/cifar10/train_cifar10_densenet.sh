#!/bin/bash
# sh scripts/basics/cifar10/train_cifar10_densenet.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/cifar10/cifar10.yml \  # dataset location
configs/preprocessors/base_preprocessor.yml \ # preprocessor file location
configs/networks/densenet.yml \ # location to network to be used
configs/pipelines/train/baseline.yml \ # location to pipeline ot be used
