from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure
)
from torch.nn import MSELoss
from tqdm import tqdm

from utils.function import load_image, save_image, save_metrics, to_categorical
from utils.metrics import Evaluator

def dehazing_validation(path, model, test_datasets, device, module, type_data):
    model.eval()

    criterion = nn.MSELoss().to(device)
    criterion_psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    criterion_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    test_score = {
        'l2':[],
        'psnr':[],
        'ssim':[],
        'time':[]
    }
    
    for hazy, target in tqdm(test_datasets):
        # Load
        input = load_image(hazy)
        target = load_image(target)
        
        # Load to device
        input = input[None, ...].to(device)
        target = target[None, ...].to(device)
        
        # Feed into model
        with torch.no_grad():
            start = time()
            preds = model(input)
            test_score['time'].append(time() - start)
        
        test_score['l2'].append(criterion(preds, target).cpu().data.numpy().item())
        
        target = torch.clamp(target, min=0, max=1)
        preds = torch.clamp(preds, min=0, max=1)
        
        test_score['psnr'].append((criterion_psnr(preds, target) * target.size(0)).cpu().data.numpy().item())
        test_score['ssim'].append((criterion_ssim(preds, target) * target.size(0)).cpu().data.numpy().item())


    save_metrics(
        path,
        metrics={
            "psnr":list(test_score['psnr']),
            "ssim":list(test_score['ssim']),
            "l2":list(test_score['l2'])
        }, 
        model=module,
        type_data=type_data
    )
        
    return test_score


def segmentation_metric(default_path, metrics_path, model, datasets, n_class, device, colour_codes, module="MTLPretrained", is_hazy=True):
    metrics_func = Evaluator(num_class=n_class, device=device)
    div_metrics_func = Evaluator(num_class=n_class, device=device)
    
    test_score = {
        'l2':[],
        'psnr':[],
        'ssim':[],
        'miou':[],
        'dice':[],
        'iou':[],
        'oa':[],
        'times':[]
    }
    metrics_func.reset()
    
    for index, i in enumerate(tqdm(datasets)):
        div_metrics_func.reset()
        
        # Load Dataset
        x = load_image(i[0 if is_hazy else 1])
        y = load_image(i[2], False)
        
        y = to_categorical(y, colour_codes)
        y = np.argmax(y, axis=-1)
        y = torch.as_tensor(y).to(torch.long)

        x = x[None, ...].to(device)
        y = y[None,...].to(device)
        
        
        # Predict
        with torch.no_grad():
            start = time()
            pred = model(x)
            test_score['times'].append(time()-start)
            
        pred = softmax(pred, dim=1)

        metrics_func.add_batch(
            y, 
            pred.argmax(dim=1)
        )

        div_metrics_func.add_batch(
            y, 
            pred.argmax(dim=1)
        )

        # Stored Metrics
        test_score['miou'].append(div_metrics_func.meanIntersectionOverUnion().cpu().data.numpy().item())
        test_score['oa'].append(div_metrics_func.OA().cpu().data.numpy().item())
        test_score['dice'].append(np.nanmean(div_metrics_func.F1().cpu().data.numpy()).item())

        if index < 30:
            save_image(
                image=colour_codes[pred.argmax(dim=1).cpu().detach().numpy()[0]].astype(np.uint8), 
                path=i[0], 
                save_path=default_path, 
                method=module,
                is_hazy=is_hazy,
                mode="segmentation"
            )

    save_metrics(
        path=metrics_path,
        metrics={
            "times":list(test_score['times']),
            "miou":list(test_score['miou']),
            "oa":list(test_score['oa']),
            "dice":list(test_score['dice']),
        }, 
        model=module,
        type_data="hazy" if is_hazy else "clear"
    )
    
    test_score['miou'] = metrics_func.meanIntersectionOverUnion().cpu().data.numpy().item()
    test_score['dice'] = metrics_func.F1().cpu().data.numpy()
    test_score['iou'] = metrics_func.Intersection_over_Union().cpu().data.numpy()
    test_score['oa'] = metrics_func.OA().cpu().data.numpy().item()
    test_score['times'] = np.mean(np.asarray(test_score['times']))

    return test_score
    
def evaluation_metric(default_path, metrics_path, model, datasets, n_class, device, colour_codes, module="MTLPretrained", is_hazy=True, activation=softmax, reverse=False):
    criterion = MSELoss().to(device)
    criterion_psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    criterion_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    metrics_func = Evaluator(num_class=n_class, device=device)
    div_metrics_func = Evaluator(num_class=n_class, device=device)
    
    test_score = {
        'l2':[],
        'psnr':[],
        'ssim':[],
        'miou':[],
        'dice':[],
        'iou':[],
        'oa':[],
        'times':[]
    }
    metrics_func.reset()
    
    for index, i in enumerate(tqdm(datasets)):
        div_metrics_func.reset()
        
        # Load Dataset
        x = load_image(i[0 if is_hazy else 1])
        y_rec = load_image(i[1])
        y = load_image(i[2], False)
        
        y = to_categorical(y, colour_codes)
        y = np.argmax(y, axis=-1)
        y = torch.as_tensor(y).to(torch.long)

        x = x[None, ...].to(device)
        y_rec = y_rec[None, ...].to(device)
        y = y[None,...].to(device)
        
        # Predict
        with torch.no_grad():
            start = time()

            if reverse:
                pred, recons = model(x)
            else:
                recons, pred = model(x)
            test_score['times'].append(time()-start)

        if activation != None:
            pred = softmax(pred, dim=1)

        metrics_func.add_batch(
            y, 
            pred.argmax(dim=1)
        )
        
        div_metrics_func.add_batch(
            y, 
            pred.argmax(dim=1)
        )

        test_score['l2'].append(criterion(recons, y_rec).cpu().data.numpy().item())
        
        # Normalize [0, 1]
        recons = torch.clamp(recons, 0, 1)
        y_rec = torch.clamp(y_rec, 0, 1)

        test_score['psnr'].append(np.nan_to_num(criterion_psnr(recons, y_rec).cpu().data.numpy()).item())
        test_score['ssim'].append(np.nan_to_num(criterion_ssim(recons, y_rec).cpu().data.numpy()).item())

        test_score['miou'].append(np.nan_to_num(div_metrics_func.meanIntersectionOverUnion().cpu().data.numpy()).item())
        test_score['oa'].append(np.nan_to_num(div_metrics_func.OA().cpu().data.numpy()).item())
        test_score['dice'].append(np.nan_to_num(np.nanmean(div_metrics_func.F1().cpu().data.numpy())).item())

        if index < 30:
            save_image(
                image=colour_codes[pred.argmax(dim=1).cpu().detach().numpy()[0]].astype(np.uint8), 
                path=i[0], 
                save_path=default_path, 
                method=module,
                is_hazy=is_hazy,
                mode="segmentation"
            )

            save_image(
                image=recons[0].cpu().detach().numpy().transpose(1, 2, 0), 
                path=i[0], 
                save_path=default_path, 
                method=module,
                is_hazy=is_hazy,
                mode="reconstruction"
            )
            
    save_metrics(
        path=metrics_path,
        metrics={
            "psnr":list(test_score['psnr']),
            "ssim":list(test_score['ssim']),
            "l2":list(test_score['l2']),
            "times":list(test_score['times']),
            "miou":list(test_score['miou']),
            "oa":list(test_score['oa']),
            "dice":list(test_score['dice']),
        }, 
        model=module,
        type_data="hazy" if is_hazy else "clear"
    )

    test_score['miou'] = metrics_func.meanIntersectionOverUnion().cpu().data.numpy()
    test_score['dice'] = metrics_func.F1().cpu().data.numpy()
    test_score['iou'] = metrics_func.Intersection_over_Union().cpu().data.numpy()
    test_score['oa'] = metrics_func.OA().cpu().data.numpy()
    test_score['times'] = np.mean(np.asarray(test_score['times']))
    test_score['psnr'] = np.asarray(np.mean(test_score['psnr']))
    test_score['ssim'] = np.asarray(np.mean(test_score['ssim']))

    return test_score