# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import functools
import os
import subprocess
import tensorflow as tf
#import horovod.tensorflow as hvd
import herring.tensorflow as herring

def init_dist():
    herring.init()
    # tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[herring.local_rank()], 'GPU')

def get_dist_info():
    #print("here we got the info {}, size is {}".format(herring.local_size(), herring.size()))
    return herring.rank(), herring.local_rank(), herring.size(), int(herring.local_size()/2) #TODO return a dict instead

def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _, _, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper

def broadcast_weights(runner):
    print('Rank {} broadcasting'.format(runner.rank))
    herring.broadcast_variables(runner.model.variables, root_rank=0)
    #hvd.broadcast_variables(runner.optimizer.variables(), root_rank=0)
    print('Variable broadcast done.')

def get_distributed_tape(tape):
    return herring.DistributedGradientTape(tape, device_dense='/gpu:0')
    '''return herring.DistributedGradientTape(tape,
                device_dense='/gpu:0',
                device_sparse='',
                # compression=hvd.Compression.fp16, # hurts convergence in 8x8 case
                compression=herring.Compression.none,
                sparse_as_dense=False)'''

def get_barrier():
    #print("we're here for barrier")
    print("The rank is {}".format(herring.rank()))
    a = tf.constant(0, dtype=tf.float32)
    #sess = tf.compat.v1.Session(config=tf.ConfigProto(log_device_placement=True))
    #print(sess.run(a))
    return herring.oob_allreduce(a)