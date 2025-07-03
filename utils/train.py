# RSHazeNet
from collections import defaultdict
import gc
import json
import os
import time
from torch.cuda.amp import autocast
import torch
from torch.nn import MSELoss
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from torchmetrics.image import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)



# Fungsi loss yang digunakan sekaligus update state loss dan metrics
def calc_loss(criterion, pred, target, metrics):
    loss = criterion['mse'](pred, target)
    
    target = torch.clamp(target, min=0, max=1)
    pred = torch.clamp(pred, min=0, max=1)
    
    metrics['ssim'] += criterion['ssim'](pred, target).cpu().data.numpy() * target.size(0)
    metrics['psnr'] += criterion['psnr'](pred, target).cpu().data.numpy() * target.size(0)
    metrics['loss'] += loss.cpu().data.numpy() * target.size(0)
    metrics['l1'] += loss.cpu().data.numpy() * target.size(0)
    return loss


# Print metric per epoch
def print_metrics(metrics, global_metrics, epoch_samples, phase, epoch, phase_time):    
    outputs = []
    
    for k in metrics.keys():
        outputs.append("{}: {}".format(k, metrics[k] / epoch_samples))
        global_metrics[phase][k].append(metrics[k] / epoch_samples)
        
        
    global_metrics[phase]['epoch'] = epoch
    global_metrics[phase]['time'].append(phase_time)
    
    print("{}: {}".format(phase, ", ".join(outputs)))
    
    return global_metrics

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_model(dataloaders, phase_list, model, optimizer_lr=1e-3, num_epochs=1000, checkpoint_path='/content', resume=False, device='cpu'):
    # Inisialisasi loss
    criterion = {
        'mse':MSELoss().to(device),
        'psnr':PeakSignalNoiseRatio(data_range=1.0).to(device),
        'ssim':StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    }

    # Inisialisasi metrics
    global_metrics = {}
    
    for i in phase_list:
        global_metrics[i] = {
            'time':[],
            'epoch':[],
            'loss':[],
            'psnr':[],
            'ssim':[],
            'l1':[]
        }
    
    # Inisialisasi opt
    optimizer = optim.Adam(model.parameters(), lr=optimizer_lr)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-8)
    scaler = torch.cuda.amp.GradScaler()
    
    # Inisialisasi loss threshold
    best_loss, best_epoch, best_model, last_epoch = 0, 0, 0, 0
    
    # Resume Checkpoint
    if os.path.exists(f'{checkpoint_path[0].split("pth")[0]}_checkpoint.pth') and resume:
        with open(f"{checkpoint_path[0].split('pth')[0]}json", "r") as f: 
            global_metrics = json.load(f)
            
        last_epoch = len(global_metrics['train']['loss'])
        print("Maximum PSNR: ", max(global_metrics["val"]["psnr"]))
        best_epoch = torch.load(
            f'{checkpoint_path[0].split("pth")[0]}_best.pth', 
            map_location=torch.device(device)
        )["best_epoch"]
        
        checkpoint = torch.load(
            f'{checkpoint_path[0].split("pth")[0]}_checkpoint.pth', 
            map_location=torch.device(device)
        )
        best_loss = global_metrics["val"]["psnr"][best_epoch]
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model.load_state_dict(checkpoint['model'])
        
        print(f"Load Checkpoint: {checkpoint['epoch']}")

    print("Init Best Loss: ", best_loss)
    for epoch in range(last_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in phase_list:
            phase_time = time.time()
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            # Inisialisasi metric dict guna menyimpan hasil pembelajaran
            metrics = defaultdict(float)
            epoch_samples = 0

            for index, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):

                inputs = inputs.to(device)
                labels = labels.to(device)
                
                if phase == 'train':
                    inputs = torch.autograd.Variable(inputs)
                    labels = torch.autograd.Variable(labels)
                
                # reset gradient parameter
                model.zero_grad()
                optimizer.zero_grad()
                  
                # forward pass
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    with autocast(True):
                        # Generator Training
                        outputs = model(inputs)
                        loss = calc_loss(criterion, outputs, labels, metrics)
                    
                    # backpropagation in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update() 
                        
                # statistics
                epoch_samples += inputs.size(0)
                
                del inputs, labels, outputs
                gc.collect()
                
            phase_time = time.time() - phase_time
            global_metrics = print_metrics(metrics, global_metrics, epoch_samples, phase, epoch, phase_time)
            
            if phase == 'train':
              scheduler.step()
              for param_group in optimizer.param_groups:
                  print("LR G", param_group['lr'])

            # fase val utk testing dan saving weights
            if phase =='val':
                if (epoch+1) % 50 == 0 or epoch == 0:
                    image, ground_truth = next(iter(dataloaders['test']))

                    with torch.no_grad():
                        prediction = model(image[:1].to(device))
                        
                    prediction = torch.clamp(prediction, min=0, max=1)
                    image = torch.clamp(image, min=0, max=1)
                    ground_truth = torch.clamp(ground_truth, min=0, max=1)
                    
                    prediction = prediction[0].cpu().detach().numpy().transpose((1, 2, 0))
                    image = image[0].numpy().transpose((1, 2, 0))
                    ground_truth = ground_truth[0].numpy().transpose((1, 2, 0))
                    
                    plt.figure(figsize=(4*3, 4))
                    ax1 = plt.subplot(1, 3, 1)
                    ax1.imshow(image)
                    ax1.set_title('Image')
                    ax1.axis('off')

                    ax2 = plt.subplot(1, 3, 2)
                    ax2.imshow(prediction)
                    ax2.set_title('Prediction')
                    ax2.axis('off')

                    ax3 = plt.subplot(1, 3, 3)
                    ax3.imshow(ground_truth)
                    ax3.set_title('GT')
                    ax3.axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                    
                    del image, ground_truth, prediction
                    gc.collect()
                    
                epoch_loss = metrics['psnr'] / epoch_samples

                if ((epoch == 0) or (epoch_loss >= best_loss)):
                    print(f"saving best model to {checkpoint_path} | Epoch: {best_epoch}")
                    best_loss = epoch_loss
                    best_epoch = epoch
                    
                    checkpoint = { 
                        'epoch': epoch,
                        'best_epoch':best_epoch,
                        'model':model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }
                    torch.save(checkpoint, f'{checkpoint_path[0].split("pth")[0]}_best.pth')

                        
            # Metrics to json
            with open(f"{checkpoint_path[0].split('pth')[0]}json", "w") as f: 
                json.dump(global_metrics, f)
            
            time_elapsed = time.time() - since
            print('Phase {}: {:.0f}m {:.0f}s'.format(phase, time_elapsed // 60, time_elapsed % 60))

        checkpoint = { 
            'epoch': epoch,
            'model':model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(checkpoint, f'{checkpoint_path[0].split("pth")[0]}_checkpoint.pth')
        
        time_elapsed = time.time() - since
        print('Total: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
    print('Best val loss: {:4f}'.format(best_loss), 'Best Epoch:', best_epoch)