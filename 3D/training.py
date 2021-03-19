#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import os
import numpy as np
from tqdm.autonotebook import tqdm

from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True

def train_epoch(model, dataloader, optimizer, loss_fn, finetune_fn=None, semantic_model=None):
    model.train()
    misclassification_percentage = 0
    epoch_train_loss = 0

    for meta_data, indices in dataloader:
        for key in meta_data:
            meta_data[key] = meta_data[key].cuda()
        prediction, params = model(meta_data)

        # Do normal training of the shape prediction.
        if finetune_fn is None:
            batch_size, num_query_points, channels = prediction.shape
            batch_loss = loss_fn(prediction, meta_data['query_y'],
                                 model.module.sigma_outer)
            pred_sign = (prediction[...,0] > 0.5).float() if channels == 1 \
                        else torch.sign(prediction[...,0])
            gt_sign = torch.sign(meta_data['query_y'][...,-1])
            misclassification_percentage += (pred_sign != gt_sign).float().mean().detach().cpu().item()
            epoch_train_loss += batch_loss.item()

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
            optimizer.step()
        # Finetune the classification layer on the features of the successfully
        # adapted network.
        else:
            from ..modules import MetaFCNet
            # Use params to get feature vectors for the points in the query
            #hyperparams = model.module.hypo_module.hyperparams
            hyperparams = dict(in_features=2,
                                out_features=2,
                                num_hidden_layers=8,
                                hidden_features=128,
                                positional_encoding=True,
                                skip_connect=False)

            feature_extractor = MetaFCNet(**hyperparams, features_only=True)
            features = feature_extractor.forward(meta_data['query_x'], params=params)
            loss = finetune_fn(semantic_model, features, meta_data['semantic_labels'])
            epoch_train_loss += loss
        
    misclassification_percentage /= len(dataloader)
    epoch_train_loss /= len(dataloader)
    
    return epoch_train_loss, misclassification_percentage
    
def val_epoch(model, dataloader, loss_fn, finetune_fn=None, semantic_model=None):
    misclassification_percentage = 0
    epoch_loss = 0

    model.eval()
    for meta_data, indices in dataloader:
        with torch.no_grad():
            for key in meta_data:
                meta_data[key] = meta_data[key].cuda()
            prediction, params = model(meta_data)

            # Do normal validation of the shape prediction.
            if finetune_fn is None:
                batch_size, num_query_points, channels = prediction.shape
                batch_loss = loss_fn(prediction, meta_data['query_y'], model.module.sigma_outer)
                pred_sign = (prediction[...,0] > 0.5).float() if channels == 1 else torch.sign(prediction[...,0])
                gt_sign = torch.sign(meta_data['query_y'][...,-1])
                misclassification_percentage += (pred_sign != gt_sign).float().mean().detach().cpu().item()
                epoch_loss += batch_loss.item()
            # Finetune the classification layer on the features of the successfully
            # adapted network.
            else:
                from ..modules import MetaFCNet
                # Use params to get feature vectors for the points in the query
                #hyperparams = model.module.hypo_module.hyperparams
                hyperparams = dict(in_features=2,
                                    out_features=2,
                                    num_hidden_layers=8,
                                    hidden_features=128,
                                    positional_encoding=True,
                                    skip_connect=False)

                feature_extractor = MetaFCNet(**hyperparams, features_only=True)
                features = feature_extractor.forward(meta_data['query_x'], params=params)
                loss = finetune_fn(semantic_model, features, meta_data['semantic_labels'], val=True)
                epoch_loss += loss


    misclassification_percentage /= len(dataloader)
    epoch_loss /= len(dataloader)
    return epoch_loss, misclassification_percentage

def train(model, optimizer, scheduler, dataloader, start_epoch, num_epochs, training_mode, output_dir='', save_freq=100, val_dataloader=None, loss_fn=None, finetune_fn=None, semantic_model=None):
    writer = SummaryWriter(output_dir)
        
    for epoch in tqdm(range(start_epoch, num_epochs + 1)):
        
        epoch_train_loss, epoch_train_misclassification_percentage = train_epoch(model, dataloader, optimizer, loss_fn, finetune_fn, semantic_model)
        epoch_val_loss, epoch_val_misclassification_percentage = val_epoch(model, val_dataloader, loss_fn, finetune_fn, semantic_model)
        
        scheduler.step()


        tqdm.write(f"Epoch: {epoch} \t Train Loss: {epoch_train_loss:.4f} \t Train Misclassified %: {epoch_train_misclassification_percentage*100:.2f} \t Val Loss: {epoch_val_loss:.4f} \t Val Misclassified %: {epoch_val_misclassification_percentage*100:.2f}\t {output_dir}")
        
        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/Val', epoch_val_loss, epoch)
        writer.add_scalar('Misclassified_Percentage/Train', epoch_train_misclassification_percentage, epoch)
        writer.add_scalar('Misclassified_Percentage/Val', epoch_val_misclassification_percentage, epoch)

        if semantic_model:
            torch.save({'model': semantic_model}, os.path.join(output_dir, 'semantic_latest.pth'))
            
        torch.save({'model': model,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch}, os.path.join(output_dir, 'latest.pth'))
        if epoch % save_freq == 0:
            torch.save({'model': model,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch}, os.path.join(output_dir, f'{epoch:04d}.pth'))
            if semantic_model:
                torch.save({'model': semantic_model}, os.path.join(output_dir, f'semantic_{epoch:04d}.pth'))

