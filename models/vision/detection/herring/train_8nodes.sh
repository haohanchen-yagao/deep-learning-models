# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash
NUM_GPU=8
TRAIN_CFG=$1

echo ""
echo "NUM_GPU: ${NUM_GPU}"
echo "TRAIN_CFG: ${TRAIN_CFG}"
echo ""

cd /shared/deep-learning-models/models/vision/detection
export PYTHONPATH=${PYTHONPATH}:${PWD}

herringrun -n 8 \
-c /shared/tf22envs \
--homogeneous \
-x LD_LIBRARY_PATH \
-x PATH \
-x NCCL_SOCKET_IFNAME=^docker0,lo \
-x NCCL_DEBUG=INFO \
-x TF_CUDNN_USE_AUTOTUNE=0 \
-x PYTHONPATH \
python tools/train.py \
--config ${TRAIN_CFG} \
--autoscale-lr \
--amp
