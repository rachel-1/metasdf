#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
import torch
import sys
from . import levelset_data

torch.backends.cudnn.benchmark = True

from meta_modules import *
from modules import *

def reconstruct(decoder, data, mesh_filename, context_mode, max_batch=2000000, N=256):
    try:
        reconstructed_sdf = generate_dense_cube(decoder, data, context_mode, max_batch, N)
    except OSError as e:
        print("OS Error: ", e)
        return
    
    if not os.path.exists(os.path.dirname(mesh_filename)):
        os.makedirs(os.path.dirname(mesh_filename))

    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    levelset_data.convert_sdf_samples_to_ply(
        reconstructed_sdf.data,
        voxel_origin,
        voxel_size,
        mesh_filename,
        offset=data['norm_params']['offset'],
        scale=data['norm_params']['scale'],
        level=0.0
    )

def generate_dense_cube(decoder, sampled_data, context_mode, max_batch, N, test_time_optim_steps=None):
    if context_mode == 'partial':
        meta_data = levelset_data.meta_split(sampled_data['sdf'].unsqueeze(0),
                                             sampled_data['partial'].unsqueeze(0),
                                             context_mode=context_mode)
    else:
        meta_data = levelset_data.meta_split(sampled_data['sdf'].unsqueeze(0),
                                             sampled_data['levelset'].unsqueeze(0),
                                                 context_mode=context_mode)

    ####### Use the given SDF samples as context to adapt the meta-network ##########
    context_x = meta_data['context'][0].cuda()
    context_y = meta_data['context'][1].cuda()

    with torch.no_grad():
        start_time = time.time()
        params = decoder.generate_params(context_x, context_y, intermediate=False, num_meta_steps=test_time_optim_steps)

        print(f"Adaptation in {time.time() - start_time} seconds")

        ###### Reconstruct mesh by sampling densely from a 256^3 cube ###########
        reconstruction_points = levelset_data.create_samples(N=N).cuda()
        reconstructed_sdf = torch.zeros((reconstruction_points.shape[0], 1)).cpu()

        decoder.eval()
        head = 0
        while head < reconstruction_points.shape[0]:
            query_x = reconstruction_points[head:min(head + max_batch, reconstruction_points.shape[0]), 0:3].unsqueeze(0)

            # When reconstructing intermediate steps, loop through parameters of each iteration.
            predictions = decoder.forward_with_params(query_x, params).detach()
            predictions = predictions.squeeze(0)

            if predictions.shape[-1] == 2:
                sdf = torch.sign(predictions[:, 0:1]) * torch.abs(predictions[:, 1:2]) # If using composite loss
#                 sdf = predictions[:, 1:2]
            else:
                sdf = predictions

            reconstructed_sdf[head:min(head + max_batch, reconstruction_points.shape[0]), 0] = sdf.squeeze(1).detach().cpu()
            head += max_batch
    reconstructed_sdf = reconstructed_sdf.reshape(N, N, N)
    return reconstructed_sdf
