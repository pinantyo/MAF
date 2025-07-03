# Fungsi loss yang digunakan sekaligus update state loss dan metrics
from collections import defaultdict
import gc
import json
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.nn import MSELoss, CrossEntropyLoss
from utils.metrics import Evaluator
from torchmetrics.image import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)


def calc_loss(criterion, pred, gt_clear, segments, target, metrics, sigma):
    loss_seg = criterion['ce'](segments, target)
    loss_rec = criterion['mse'](pred, gt_clear)

    sigma_rec = torch.exp(-sigma[0])
    loss_rec = torch.sum(sigma_rec * loss_rec + (sigma[0] * sigma[0]), -1)

    sigma_seg = torch.exp(-sigma[1])
    loss_seg = torch.sum(sigma_seg * loss_seg + (sigma[1] * sigma[1]), -1)

    loss = torch.mean(loss_rec + loss_seg)

    pred = torch.clamp(pred, min=0, max=1)
    gt_clear = torch.clamp(gt_clear, min=0, max=1)
    
    metrics['rec_weight'] += sigma[0].cpu().data.numpy().item() * target.size(0)
    metrics['seg_weight'] += sigma[1].cpu().data.numpy().item() * target.size(0)
    
    metrics['crossentropy'] += loss_seg.cpu().data.numpy() * target.size(0)
    metrics['mse'] += loss_rec.cpu().data.numpy() * target.size(0)
    metrics['loss'] += loss.cpu().data.numpy().item() * target.size(0)
    metrics['ssim'] += criterion['ssim'](pred, gt_clear).cpu().data.numpy() * target.size(0)
    metrics['psnr'] += criterion['psnr'](pred, gt_clear).cpu().data.numpy() * target.size(0)

    return loss

# Print metric per epoch
def print_metrics(metrics, global_metrics, epoch_samples, phase, metrics_func, epoch, phase_time):    
    outputs = []
    
    metrics['oa'] = torch.nanmean(metrics_func.OA()).cpu().data.numpy().item() * epoch_samples
    metrics['miou'] = torch.nanmean(metrics_func.meanIntersectionOverUnion()).cpu().data.numpy().item() * epoch_samples
    metrics['f1'] = torch.nanmean(metrics_func.F1()).cpu().data.numpy().item() * epoch_samples
    
    for k in metrics.keys():
        outputs.append("{}: {}".format(k, metrics[k] / epoch_samples))
        global_metrics[phase][k].append(metrics[k] / epoch_samples)
        
    global_metrics[phase]['epoch'] = epoch
    global_metrics[phase]['time'].append(phase_time)
        
    print("{}: {}".format(phase, ", ".join(outputs)))
    
    return global_metrics

def train_model(dataloaders, model, device, semantic_color, phase_lists, n_class=6, optimizer_lr=1e-3, num_epochs=100, checkpoint_path='/content', resume=False, visualize_factor=20):
    # Init metrics
    criterion = {
        'mse':MSELoss().to(device),
        'ce':CrossEntropyLoss().to(device),
        'psnr':PeakSignalNoiseRatio(data_range=1.0).to(device),
        'ssim':StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    }
    
    metrics_func = Evaluator(num_class=n_class, device=device).to(device)

    # Init global log
    global_metrics = {}
    
    for i in phase_lists:
        global_metrics[i] = {
            'epoch':[],
            'time':[],
            'crossentropy':[],
            'mse':[],
            'loss':[],
            'ssim':[],
            'psnr':[],
            'oa':[],
            'miou':[],
            'f1':[],
            'seg_weight':[],
            'rec_weight':[]
        }
    
    # Init opt
    optimizer = optim.Adam(
        model.parameters(), 
        lr=optimizer_lr,
        # betas=(0.5, 0.999)
    )
    
    # Init Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        num_epochs, 
        eta_min=optimizer_lr * 0.01, 
        last_epoch=-1
    )

    # Init loss threshold
    best_loss, best_dice, best_epoch, last_epoch = 0, 0, 0, 0
    color_space = np.array(semantic_color)
    
    # Resume checkpoint
    if resume:
        checkpoint = f'{checkpoint_path.split("pth")[0]}_checkpoint.pth'
        logger = f"{checkpoint_path.split('pth')[0][:-1]}.json"
        
        with open(logger, "r") as f: 
            global_metrics = json.load(f)
            
        last_epoch = len(global_metrics['train']['loss'])
        checkpoints = torch.load(checkpoint, map_location=torch.device(device))
        
        optimizer.load_state_dict(checkpoints['optimizer'])
        scheduler.load_state_dict(checkpoints['scheduler'])
        model.load_state_dict(checkpoints['model'])
        best_loss = checkpoints['best_loss']
        best_dice = checkpoints['best_dice']
        print("best_loss", best_loss, "best_dice", best_dice, "Weights Loaded")
    
    for epoch in range(last_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in phase_lists:
            metrics_func.reset()
            phase_time = time.time()
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
         
            # Init metric dict for logging
            metrics = defaultdict(float)
            epoch_samples = 0

            for index, (inputs_haze, gt_clear, labels) in enumerate(tqdm(dataloaders[phase])):
                inputs_haze = inputs_haze.to(device)
                gt_clear = gt_clear.to(device)
                labels = labels.to(device)

                # reset gradient parameter
                model.zero_grad()
                optimizer.zero_grad()

                # forward pass
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, segments, sigma = model(inputs_haze, True)
                    loss = calc_loss(criterion, outputs, gt_clear, segments, labels.long(), metrics, sigma)
                    
                    # backpropagation in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
 
                    segments = F.softmax(segments, dim=1).argmax(dim=1)
                    metrics_func.add_batch(
                        labels, 
                        segments
                    )

                # statistics
                epoch_samples += inputs_haze.size(0)
                
                del inputs_haze, gt_clear, labels, outputs, segments
                gc.collect()

            phase_time = time.time() - phase_time
            global_metrics = print_metrics(metrics, global_metrics, epoch_samples, phase, metrics_func, epoch, phase_time)
            
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                
            metrics_func.reset()
            
            # save checkpoints & visualize samples
            if phase =='val':
                epoch_loss = metrics['loss'] / epoch_samples
                epoch_dice = global_metrics['val']['f1'][-1]
                
                if (epoch+1) % visualize_factor == 0 or epoch == 0:
                    image_hazy, image_clear, ground_truth = next(iter(dataloaders['test']))
                    
                    with torch.no_grad():
                        prediction, segments = model(image_hazy.to(device), False)
                        
                    segments = F.softmax(segments, dim=1).argmax(dim=1)
                    
                    metrics_func.add_batch(
                        ground_truth.to(device),
                        segments
                    )
                    
                    dice_score = torch.nanmean(metrics_func.F1()).cpu().data.numpy().item()    
                    iou_score = metrics_func.meanIntersectionOverUnion().cpu().data.numpy().item()
                    
                    image_hazy = image_hazy.cpu().detach().numpy()[0].transpose((1, 2, 0))
                    image_clear = image_clear.cpu().detach().numpy()[0].transpose((1, 2, 0))
                    prediction = torch.clamp(prediction, 0, 1)
                    prediction = prediction.cpu().detach().numpy()[0].transpose((1, 2, 0))
                    
                    ground_truth = ground_truth.cpu().detach().numpy()[0]
                    segments = segments.cpu().detach().numpy()[0]
                    
                    plt.figure(figsize=(4*5, 4))
                    plt.suptitle(f'Dice: {dice_score}, mIoU: {iou_score}')

                    ax0 = plt.subplot(1, 5, 1)
                    ax0.imshow(image_hazy)
                    ax0.set_title('Haze Image')
                    ax0.axis('off')
                    
                    ax1 = plt.subplot(1, 5, 2)
                    ax1.imshow(prediction)
                    ax1.set_title('Recons Image')
                    ax1.axis('off')
                    
                    ax12 = plt.subplot(1, 5, 3)
                    ax12.imshow(image_clear)
                    ax12.set_title('Recons Image')
                    ax12.axis('off')
                    
                    ax2 = plt.subplot(1, 5, 4)
                    ax2.imshow(color_space[segments])
                    ax2.set_title('Prediction Mask')
                    ax2.axis('off')

                    ax3 = plt.subplot(1, 5, 5)
                    ax3.imshow(color_space[ground_truth])
                    ax3.set_title('GT Mask')
                    ax3.axis('off')
                    plt.show()
                    
                    del image_hazy, image_clear, ground_truth, prediction, segments
                    gc.collect()

                if ((epoch == 0) or (epoch_loss < best_loss)):
                    best_loss = epoch_loss
                    print(f"saving best model to {checkpoint_path} best loss")
                    checkpoint = { 
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }
                    print(f'Save checkpoint {checkpoint_path.split("pth")[0]}_bestloss.pth')
                    torch.save(checkpoint, f'{checkpoint_path.split("pth")[0]}_bestloss.pth')

                if ((epoch == 0) or (epoch_dice > best_dice)):
                    best_dice = epoch_dice
                    print(f"saving best model to {checkpoint_path} best dice")
                    checkpoint = { 
                        'epoch': epoch,
                        'best_loss': best_dice,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }
                    print(f'Save checkpoint {checkpoint_path.split("pth")[0]}_bestdice.pth')
                    torch.save(checkpoint, f'{checkpoint_path.split("pth")[0]}_bestdice.pth')

                checkpoint = { 
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'best_dice':best_dice,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                print(f'Save checkpoint {checkpoint_path.split("pth")[0]}_checkpoint.pth')
                torch.save(checkpoint, f'{checkpoint_path.split("pth")[0]}_checkpoint.pth')
                
            # Metrics to json
            with open(f"{checkpoint_path.split('pth')[0][:-1]}.json", "w") as f:
                json.dump(global_metrics, f)
                
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f} | Best epoch {}'.format(best_loss, best_epoch))
    torch.cuda.empty_cache()