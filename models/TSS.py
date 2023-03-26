import torch
import pytorch_lightning as pl
from models.network import Network
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from utils.init_func import init_weight, group_weight
from torch.nn import SyncBatchNorm, BatchNorm2d
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.classification import MultilabelJaccardIndex, MultilabelF1Score
from torchmetrics import MetricCollection
import torchvision
import wandb
from utils import DatasetDownloader, Config

class WarmupPolyScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, max_iters, power=0.9):
        self.warmup_steps = warmup_steps
        self.max_iters = max_iters
        self.power = power
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            lr = float(step) / float(max(1, self.warmup_steps))
        else:
            lr = (1 - float(step - self.warmup_steps) / float(max(1, self.max_iters - self.warmup_steps))) ** self.power
        return lr

class TSS(pl.LightningModule):
    def __init__(self, lr, init_weight_lr, init_weight_momentum, download_config):
        super().__init__()
        self.lr = lr
        self.num_classes = 6
        self.class_labels = ["bg", "farmland", "water", "forest", "structure", "meadow"]
        self.automatic_optimization = False
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)

        DatasetDownloader(**download_config).download()
        self.BatchNorm = BatchNorm2d#SyncBatchNorm if self.device.type == 'gpu' else BatchNorm2d
        self.network = Network(6,
                    pretrained_model="data/pytorch-weight/resnet50_v1c.pth",
                    norm_layer=self.BatchNorm)
        self.named_children = self.network.named_children
        self.pred_images = []

        init_weight(self.network.branch1.business_layer, nn.init.kaiming_normal_,
            BatchNorm2d, init_weight_lr, init_weight_momentum,
            mode='fan_in', nonlinearity='relu')
        init_weight(self.network.branch2.business_layer, nn.init.kaiming_normal_,
            BatchNorm2d, init_weight_lr, init_weight_momentum,
            mode='fan_in', nonlinearity='relu')

        self.save_hyperparameters()
        
        # self.metrics = MetricCollection([
        #     MultilabelJaccardIndex(num_classes=6, num_labels=6),
        #     MultilabelF1Score(num_classes=6, num_labels=6),
        # ])

        self.validation_outputs = []
    
    @property
    def labels(self):
        l = {}
        for i, label in enumerate(self.class_labels):
            l[i] = label
        return l

    def forward(self, x, step=1):
        if self.training:
            return self.network(x, step)
        return self.network(x)

    def training_step(self, batch, batch_idx):

        self.optimizers()[0].zero_grad()
        self.optimizers()[1].zero_grad()

        unsup_imgs, imgs, gts = batch
        gts = gts.squeeze()
        

        _, pred_sup_l = self(imgs, step=1)
        _, pred_unsup_l = self(unsup_imgs, step=1)
        _, pred_sup_r = self(imgs, step=2)
        _, pred_unsup_r = self(unsup_imgs, step=2)

        ### cps loss ###
        pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
        pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)
        _, max_l = torch.max(pred_l, dim=1)
        _, max_r = torch.max(pred_r, dim=1)
        max_l = max_l.long()
        max_r = max_r.long()
        cps_loss = self.criterion(pred_l, max_r) + self.criterion(pred_r, max_l)
        cps_loss = cps_loss * 1.5

        ### standard cross entropy loss ###
        loss_sup = self.criterion(pred_sup_l, gts)

        loss_sup_r = self.criterion(pred_sup_r, gts)

        loss = loss_sup + loss_sup_r + cps_loss

        loss.backward()

        self.optimizers()[0].step()
        self.optimizers()[1].step()

        self.log("cps_loss", cps_loss, on_step=True, on_epoch=False, prog_bar=True, logger=False, sync_dist=True)
        self.log("loss_sup_r", loss_sup_r, on_step=True, on_epoch=False, prog_bar=True, logger=False, sync_dist=True)
        self.log("loss_sup", loss_sup, on_step=True, on_epoch=False, prog_bar=True, logger=False, sync_dist=True)

        self.log("lr", self.optimizers()[0].param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("cps_loss_", cps_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("loss_sup_r_", loss_sup_r, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("loss_sup_", loss_sup, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        imgs, gts = batch
        gts = gts.squeeze(1)

        with torch.no_grad():
            pred_sup_l = self(imgs)

            loss_sup = self.criterion(pred_sup_l, gts)

            self.log("val_loss_sup", loss_sup, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

            preds = torch.argmax(pred_sup_l, dim=1)
            
            for i in range(len(imgs)):
                img = imgs[i]
                gt = gts[i]
                pred = preds[i]

                mask_img = wandb.Image(img.permute(1,2,0).detach().cpu().numpy(), masks={
                "predictions": {
                    "mask_data": pred.detach().cpu().numpy(),
                    "class_labels": self.labels
                },
                "ground_truth": {
                    "mask_data": gt.detach().cpu().numpy(),
                    "class_labels": self.labels
                },
                })

                self.pred_images.append(mask_img)

    def on_validation_epoch_end(self):
        wandb.log({"predictions": self.pred_images})
        self.pred_images = []
        wandb.log({'branch1_weights': self.network.branch1.state_dict()})
    #     images = self.validation_outputs

    #     # Create a grid of the images
    #     grid = torchvision.utils.make_grid(images, nrow=8, normalize=False)

    #     self.loggers[0].experiment.add_image(f'val_images', grid, self.current_epoch)

    #     self.validation_outputs = []

    def test_step(self, batch, batch_idx):
        imgs, gts = batch
        gts = gts.squeeze(1)

        with torch.no_grad():
            pred_sup_l = self(imgs)
            loss_sup = self.criterion(pred_sup_l, gts)

            preds = torch.argmax(pred_sup_l, dim=1)

            # Calculate mIoU and F1 score
            miou, iou_per_class, pixel_counts = calculate_iou(preds, gts, self.num_classes)
            f1 = calculate_f1_score(preds, gts)

            self.log("test_miou", miou, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("test_0", iou_per_class[0], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("test_1", iou_per_class[1], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True) 
            self.log("test_2", iou_per_class[2], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("test_3", iou_per_class[3], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("test_4", iou_per_class[4], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("test_5", iou_per_class[5], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("test_f1", f1, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("test_loss_sup", loss_sup, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     # pred = outputs['pred']
    #     # target = outputs['target']
    #     loss = outputs['loss']
    #     # target = torch.nn.functional.one_hot(target, num_classes=6).permute(0, 3, 1, 2)

    #     # self.metrics.update(pred, target)
    #     self.log('train_loss', loss, logger=False, prog_bar=True)

    def on_train_epoch_end(self):
        self.lr_schedulers()[0].step()
        self.lr_schedulers()[1].step()

    # def on_test_epoch_end(self, outputs):
    #     test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     self.log('test_epoch_loss', test_loss)

    def configure_optimizers(self):
        params_list_l = []
        
        params_list_l = group_weight(params_list_l, self.network.branch1.backbone,
                               self.BatchNorm, self.lr)
        for module in self.network.branch1.business_layer:
            params_list_l = group_weight(params_list_l, module, self.BatchNorm, self.lr)

        optimizer_l = torch.optim.SGD(
                                params_list_l,
                                lr=self.lr,
                                momentum=0.1,
                                weight_decay=1e-4)

        params_list_r = []
        params_list_r = group_weight(params_list_r, self.network.branch2.backbone,
                               self.BatchNorm, self.lr)
        for module in self.network.branch2.business_layer:
            params_list_r = group_weight(params_list_r, module, self.BatchNorm, self.lr)

        optimizer_r = torch.optim.SGD(
                                params_list_r,
                                lr=self.lr,
                                momentum=0.1,
                                weight_decay=1e-4)
        max_steps = self.trainer.datamodule.train_dataloader().batch_size
        max_iters = self.trainer.max_epochs * max_steps
        scheduler = [
            WarmupPolyScheduler(optimizer_l, warmup_steps=0, max_iters=max_iters ),
            WarmupPolyScheduler(optimizer_r, warmup_steps=0, max_iters=max_iters ),
        ]
        return [optimizer_l, optimizer_r], scheduler

    def save_checkpoint(self, filepath):
        checkpoint = {
            'state_dict': self.state_dict(),
            'optimizers': [opt.state_dict() for opt in self.optimizers()],
            'lr_schedulers': [sched.state_dict() for sched in self.lr_schedulers()]
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['state_dict'])
        for i, opt in enumerate(self.optimizers()):
            opt.load_state_dict(checkpoint['optimizers'][i])
        for i, sched in enumerate(self.lr_schedulers()):
            sched.load_state_dict(checkpoint['lr_schedulers'][i])


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

from sklearn.metrics import f1_score
def calculate_f1_score(pred, target):
    # Calculate F1 score
    f1 = f1_score(target.flatten().cpu(), pred.flatten().cpu(), average='macro')
    
    return f1