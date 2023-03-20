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

class WarmupPolyScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, max_iters, power=0.9):
        self.warmup_steps = warmup_steps
        self.max_iters = max_iters
        self.power = power
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        else:
            return (1 - float(step - self.warmup_steps) / float(max(1, self.max_iters - self.warmup_steps))) ** self.power

class TSS(pl.LightningModule):
    def __init__(self, lr=.001):
        super().__init__()
        self.lr = lr
        self.automatic_optimization = False
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)

        self.BatchNorm = BatchNorm2d#SyncBatchNorm if self.device.type == 'gpu' else BatchNorm2d
        self.network = Network(6, criterion=self.criterion,
                    pretrained_model="data/pytorch-weight/resnet50_v1c.pth",
                    norm_layer=self.BatchNorm)
        # self.named_children = self.network.named_children

        # self.metrics = MetricCollection([
        #     MultilabelJaccardIndex(num_classes=6, num_labels=6),
        #     MultilabelF1Score(num_classes=6, num_labels=6),
        # ])


    def forward(self, x, step):
        return self.network(x, step)

    def training_step(self, batch, batch_idx):

        self.optimizers()[0].zero_grad()
        self.optimizers()[1].zero_grad()

        unsup_imgs, imgs, gts = batch
        gts = gts.squeeze(1)
        

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

        # # reset the learning rate
        # self.optimizers()[0].param_groups[0]['lr'] = lr
        # self.optimizers()[0].param_groups[1]['lr'] = lr
        # for i in range(2, len(self.optimizers()[0].param_groups)):
        #     self.optimizers()[0].param_groups[i]['lr'] = lr
        # self.optimizers()[1].param_groups[0]['lr'] = lr
        # self.optimizers()[1].param_groups[1]['lr'] = lr
        # for i in range(2, len(self.optimizers()[1].param_groups)):
        #     self.optimizers()[1].param_groups[i]['lr'] = lr

        loss = loss_sup + loss_sup_r + cps_loss

        loss.backward()

        self.optimizers()[0].step()
        self.optimizers()[1].step()

        # self.log("miou", miou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        # self.log('lr', self.lr, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # return {'loss': loss, }#'pred': pred_sup_r, 'target': gts}

    # def validation_step(self, batch, batch_idx):
    #     imgs, gts = batch
    #     gts = gts.squeeze(1)

    #     with torch.no_grad():     
    #         pred_sup_l = self(imgs, step=1)
    #         # _, pred_unsup_l = self(unsup_imgs, step=1)
    #         pred_sup_r = self(imgs, step=2)
    #         # _, pred_unsup_r = self(unsup_imgs, step=2)

    #         loss_l = torch.nn.functional.cross_entropy(pred_sup_l, gts)
    #         loss_r = torch.nn.functional.cross_entropy(pred_sup_r, gts)

            # self.log('loss_l', loss_l, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            # self.log('loss_r', loss_r, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = torch.nn.functional.cross_entropy(y_hat, y)

    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     # pred = outputs['pred']
    #     # target = outputs['target']
    #     loss = outputs['loss']
    #     # target = torch.nn.functional.one_hot(target, num_classes=6).permute(0, 3, 1, 2)

    #     # self.metrics.update(pred, target)
    #     self.log('train_loss', loss, logger=False, prog_bar=True)

    # def on_train_epoch_end(self):
        # self.log("miou", self.metrics[0].compute())
        # self.log_dict(self.metrics.compute(), on_step=False, on_epoch=True)
        # self.metrics.reset()

    # def on_validation_epoch_end(self, outputs):
    #     val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     self.log('val_epoch_loss', val_loss)

    # def on_test_epoch_end(self, outputs):
    #     test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     self.log('test_epoch_loss', test_loss)

    def configure_optimizers(self):
        # params_list_l = []
        # params_list_r = []
        # params_list_l = group_weight(params_list_l, self.network.branch1.backbone,
        #                        self.BatchNorm, self.lr)
        # for module in self.network.branch1.business_layer:
        #     params_list_l = group_weight(params_list_l, module, self.BatchNorm, self.lr)
        # params_list_r = group_weight(params_list_r, self.network.branch2.backbone,
        #                        self.BatchNorm, self.lr)
        # for module in self.network.branch2.business_layer:
        #     params_list_r = group_weight(params_list_r, module, self.BatchNorm, self.lr)

        param = nn.ParameterList([
            self.network.branch1.backbone.parameters(),
            self.network.branch1.business_layer.parameters()

        ])
        optimizer_l = torch.optim.SGD(self.network.branch1.parameters(),
                                lr=self.lr,
                                momentum=0.1,
                                weight_decay=1e-4)

        param = nn.ParameterList([
            self.network.branch2.backbone.parameters(),
            self.network.branch2.business_layer.parameters()

        ])
        optimizer_r = torch.optim.SGD(self.network.branch2.parameters(),
                                lr=self.lr,
                                momentum=0.1,
                                weight_decay=1e-4)

        scheduler = [
            WarmupPolyScheduler(optimizer_l, warmup_steps=1000, max_iters=10000),
            WarmupPolyScheduler(optimizer_r, warmup_steps=1000, max_iters=10000),
        ]
        return [optimizer_l, optimizer_r]#, scheduler