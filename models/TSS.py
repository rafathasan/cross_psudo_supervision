from sklearn.metrics import f1_score
import torch
import wandb
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, StepLR, ReduceLROnPlateau, OneCycleLR
from utils.init_func import init_weight, group_weight
from torch.nn import SyncBatchNorm, BatchNorm2d
import torchvision
from utils import download_data, Config
from models.components import SingleNetwork, Head
from pytorch_lightning.loggers import WandbLogger

class TSS(pl.LightningModule):
    def __init__(self, lr, cps_scale):
        super(TSS, self).__init__()
        self.save_hyperparameters()       
        # cps scale
        # lr
        # 
        
        self.lr = lr
        self.num_classes = 6
        self.pretrained_model = "data/pytorch-weight/resnet50_v1c.pth"
        self.class_labels = ["bg", "farmland",
                             "water", "forest", "structure", "meadow"]
        self.automatic_optimization = False
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)

        if self.device.type in ['cuda', 'gpu']:
            self.BatchNorm = SyncBatchNorm
        else:
            self.BatchNorm = BatchNorm2d

        self.init_weight_lr = 1e-5
        self.init_weight_momentum = 0.9

        self.branch1 = SingleNetwork(self.num_classes, self.BatchNorm, self.pretrained_model)
        self.branch2 = SingleNetwork(self.num_classes, self.BatchNorm, self.pretrained_model)

        init_weight(self.branch1.business_layer, nn.init.kaiming_normal_,
            self.BatchNorm, self.init_weight_lr, self.init_weight_momentum,
            mode='fan_in', nonlinearity='relu')
        init_weight(self.branch2.business_layer, nn.init.kaiming_normal_,
            self.BatchNorm, self.init_weight_lr, self.init_weight_momentum,
            mode='fan_in', nonlinearity='relu')

        self.table = wandb.Table(columns=(["image", "miou", "f1"] + self.class_labels))

    @property
    def labels(self):
        l = {}
        for i, label in enumerate(self.class_labels):
            l[i] = label
        return l

    @property
    def current_lr(self):
        return self.optimizers(use_pl_optimizer=False)[0].param_groups[0]['lr']

    def forward(self, x, branch=1):
        if not self.training:
            return self.branch1(x)

        if branch == 1:
            return self.branch1(x)
        elif branch == 2:
            return self.branch2(x)

    def training_step(self, batch, batch_idx):
        
        for optimizer in self.optimizers():
            optimizer.zero_grad()

        unsup_imgs, imgs, gts = batch
        gts = gts.squeeze(1)
        
        pred_sup_l = self(imgs, branch=1)
        pred_unsup_l = self(unsup_imgs, branch=1)
        pred_sup_r = self(imgs, branch=2)
        pred_unsup_r = self(unsup_imgs, branch=2)

        ### cps loss ###
        pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
        pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)
        _, max_l = torch.max(pred_l, dim=1)
        _, max_r = torch.max(pred_r, dim=1)
        max_l = max_l.long()
        max_r = max_r.long()
        cps_loss = self.criterion(pred_l, max_r) + self.criterion(pred_r, max_l)

        ### standard cross entropy loss ###
        loss_sup = self.criterion(pred_sup_l, gts)

        loss_sup_r = self.criterion(pred_sup_r, gts)

        loss = loss_sup + loss_sup_r + cps_loss * self.hparams.cps_scale

        loss.backward()

        for optimizer in self.optimizers():
            optimizer.step()

        self.log("loss", loss_sup, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        imgs, gts = batch
        gts = gts.squeeze(1)

        pred_sup_l = self(imgs)

        loss_sup = self.criterion(pred_sup_l, gts)

        preds = torch.argmax(pred_sup_l, dim=1)

        batch_miou, batch_iou_per_class, batch_pixel_counts = calculate_iou(preds, gts, self.num_classes)
        batch_f1, batch_f1_per_class, batch_f1_pixel_counts = calculate_f1_score(preds, gts, self.num_classes)

        # self.append_to_wandb_table(imgs, gts, preds)
        
        # Log to wandb
        self.log("miou", batch_miou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("f1", batch_f1, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_loss", loss_sup, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("lr", self.current_lr, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def log_confusion_matrix(self, cm):
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.log(f'confusion_matrix/{self.class_labels[i]}_{self.class_labels[j]}', cm[i,j], on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def test_step(self, batch, batch_idx):
        from sklearn.metrics import confusion_matrix

        imgs, gts = batch
        gts = gts.squeeze(1)

        pred_sup_l = self(imgs)

        loss_sup = self.criterion(pred_sup_l, gts)

        # pred_sup_l[:, 0] = float('-inf')
        preds = torch.argmax(pred_sup_l, dim=1)

        batch_miou, batch_iou_per_class, batch_iou_pixel_counts = calculate_iou(preds, gts, self.num_classes)
        batch_f1, batch_f1_per_class, batch_f1_pixel_counts = calculate_f1_score(preds, gts, self.num_classes)

        # self.append_to_wandb_table(imgs, gts, preds)
        
        # Log to loggers
        self.log("IoU_w_bg", batch_miou, self.current_epoch)

        for k, v in enumerate(self.class_labels):
            self.log(v+"_iou", batch_iou_per_class[k], on_epoch=True)
        self.log("F1_w_bg", batch_f1, on_epoch=True)
        for k, v in enumerate(self.class_labels):
            self.log(v+"_f1", batch_f1_per_class[k], on_epoch=True)

    def on_validation_epoch_end(self):
        # for logger in self.loggers if isinstance(self.loggers, list) else [self.loggers]:
        #     if isinstance(logger, WandbLogger):
        #         logger.experiment.log({"table_summary": self.table})

        # self.table = wandb.Table(columns=(["image", "miou", "f1"] + self.class_labels))
        pass

    def on_train_epoch_end(self):
        self.lr_schedulers()[0].step()
        self.lr_schedulers()[1].step()
        pass

    def configure_optimizers(self):
        params_list_l = []
        
        params_list_l = group_weight(params_list_l, self.branch1.backbone,
                               self.BatchNorm, self.hparams.lr)
        for module in self.branch1.business_layer:
            params_list_l = group_weight(params_list_l, module, self.BatchNorm, self.hparams.lr)

        optimizer_l = torch.optim.SGD(
                                params_list_l,
                                lr=self.hparams.lr,
                                momentum=0.1,
                                weight_decay=1e-4)

        params_list_r = []
        params_list_r = group_weight(params_list_r, self.branch2.backbone,
                               self.BatchNorm, self.hparams.lr)
        for module in self.branch2.business_layer:
            params_list_r = group_weight(params_list_r, module, self.BatchNorm, self.hparams.lr)

        optimizer_r = torch.optim.SGD(
                                params_list_r,
                                lr=self.hparams.lr,
                                momentum=0.1,
                                weight_decay=1e-4)

        scheduler_l = OneCycleLR(optimizer_l, max_lr=self.hparams.lr, total_steps=30, anneal_strategy = 'cos', verbose=True)
        scheduler_r = OneCycleLR(optimizer_r, max_lr=self.hparams.lr, total_steps=30, anneal_strategy = 'cos')

        return [optimizer_l, optimizer_r], [scheduler_l, scheduler_r]
    
    def append_to_wandb_table(self, imgs, gts, preds):
        for i in range(len(imgs)):
            img = imgs[i]
            gt = gts[i]
            pred = preds[i]

            miou, iou_per_class, pixel_counts = calculate_iou(pred, gt, self.num_classes)
            f1, _, _ = calculate_f1_score(pred, gt, self.num_classes)

            img = img.permute(1, 2, 0).detach().cpu().numpy()
            gt = gt.detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()

            image_dict = wandb.Image(img, masks={
                "predictions": {
                    "mask_data": pred,
                    "class_labels": self.labels
                },
                "ground_truth": {
                    "mask_data": gt,
                    "class_labels": self.labels
                },
            })

            row = [image_dict, miou, f1] + [item for item in iou_per_class]
            # Add a row to the table
            self.table.add_data(*row)

def calculate_iou(pred, target, num_classes):
    # Initialize variables to store class-wise IoU and pixel counts
    iou_per_class = torch.zeros(num_classes)
    pixel_counts = torch.zeros(num_classes)

    # Loop over each class
    for cls in range(num_classes):
        # Get binary masks for current class
        pred_mask = (pred == cls).int()
        target_mask = (target == cls).int()

        # Check if current class is present in prediction or ground truth
        if pred_mask.sum() == 0 and target_mask.sum() == 0:
            continue

        # Calculate intersection and union
        intersection = (pred_mask * target_mask).sum()
        union = (pred_mask + target_mask).sum() - intersection

        # Calculate IoU and add to class-wise IoU and pixel counts
        iou = intersection.float() / union.float()
        iou_per_class[cls] = iou
        pixel_counts[cls] = union

    # Calculate overall mIoU
    mean_iou = iou_per_class.mean()

    return mean_iou, iou_per_class, pixel_counts


def calculate_f1_score(pred, target, num_classes):
    # Initialize variables to store class-wise F1 score and pixel counts
    f1_per_class = torch.zeros(num_classes)
    pixel_counts = torch.zeros(num_classes)

    # Loop over each class
    for cls in range(num_classes):
        # Get binary masks for current class
        pred_mask = (pred == cls).int()
        target_mask = (target == cls).int()

        # Check if current class is present in prediction or ground truth
        if pred_mask.sum() == 0 and target_mask.sum() == 0:
            continue

        # Calculate F1 score
        f1 = f1_score(target_mask.flatten().cpu(),
                      pred_mask.flatten().cpu())

        # Add F1 score and pixel counts to class-wise tensors
        f1_per_class[cls] = f1
        pixel_counts[cls] = target_mask.sum()

    # Calculate overall mF1
    mean_f1 = f1_per_class.mean()

    return mean_f1, f1_per_class, pixel_counts
