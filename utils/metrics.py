import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
class Evaluator(nn.Module):
    def __init__(self, num_class, device):
        super().__init__()
        self.device = device
        self.num_class = torch.tensor(num_class).to(self.device)
        self.confusion_matrix = torch.zeros((self.num_class,) * 2).type(torch.float32).to(self.device)
        self.eps = torch.tensor(1e-6).to(self.device)

    def get_tp_fp_tn_fn(self):
        tp = torch.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - torch.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - torch.diag(self.confusion_matrix)
        tn = torch.diag(self.confusion_matrix).sum() - torch.diag(self.confusion_matrix)
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        return precision

    def Recall(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn)
        return recall

    def F1(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp)
        Recall = tp / (tp + fn)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall)
        return F1

    def OA(self):
        OA = torch.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)
        return OA

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusion_matrix)
        union = torch.sum(self.confusion_matrix, axis=1) + torch.sum(self.confusion_matrix, axis=0) - torch.diag(
            self.confusion_matrix)
        IoU = intersection / union
        mIoU = torch.nanmean(IoU)
        return mIoU
    
    def Intersection_over_Union(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp)
        return IoU

    def Dice(self):
        intersection = torch.diag(self.confusion_matrix)
        union = torch.sum(self.confusion_matrix, axis=1) + torch.sum(self.confusion_matrix, axis=0)
        Dice = (2 * intersection + self.eps) / (union + self.eps)
        return torch.nanmean(Dice)

    def Pixel_Accuracy_Class(self):
        #         TP                                  TP+FP
        Acc = torch.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + self.eps)
        return Acc

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= torch.tensor(0).to(self.device)) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].type(torch.uint8) + pre_image[mask]
        count = torch.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape,
                                                                                                 gt_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_class,) * 2).type(torch.float32).to(self.device)