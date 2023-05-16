import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl
import wandb
import torchmetrics.functional as metrics

class DeepLabV3Plus(pl.LightningModule):
    def __init__(self, num_classes=6, lr=1e-4):
        super().__init__()
        
        # Load pre-trained ResNet50 model
        self.resnet = models.resnet50(pretrained=True)
        
        # Replace the last layer with a 1x1 convolutional layer
        self.resnet.fc = nn.Conv2d(2048, 256, 1)
        
        # DeepLabv3+ encoder
        self.aspp = ASPP(256, 256)
        
        # DeepLabv3+ decoder
        self.decoder = Decoder(256, 256, num_classes)
        
        self.lr = lr
        
        # Initialize SyncBatchNorm
        self.resnet = nn.SyncBatchNorm.convert_sync_batchnorm(self.resnet)
        self.aspp = nn.SyncBatchNorm.convert_sync_batchnorm(self.aspp)
        self.decoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.decoder)
        
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.aspp(x)
        x = self.decoder(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        
        # Forward pass
        y_pred = self.forward(x)
        
        # Compute loss
        loss = self.criterion(y_pred, y_true)
        
        # Compute IoU and F1 score
        total_miou, total_iou_per_class, total_pixel_counts = calculate_iou(y_pred_argmax, y_true, self.num_classes)
        total_f1 = calculate_f1_score(y_pred_argmax, y_true)
        
        # Log to wandb
        self.log("miou", total_miou, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("f1", total_f1, on_step=False, on_epoch=True, prog_bar=False, logger=False, sync_dist=True)
        self.log("val_loss", loss_sup, on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
        
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # ASPP with rates 1, 6, 12, and 18
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
    
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Concatenation layer
        self.concat = nn.Conv2d(out_channels * 5, out_channels, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        
        # Concatenate and apply the final 1x1 convolutional layer
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        out = self.concat(out)
        
        return out

class Decoder(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes):
        super().__init__()
        # Low-level feature fusion
        self.low_level_conv = nn.Conv2d(low_level_channels, in_channels, 1)
        
        # DeepLabv3+ decoder
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels // 4, in_channels // 8, 3, padding=1)
        self.conv4 = nn.Conv2d(in_channels // 8, num_classes, 1)
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        x, low_level_features = x
        low_level_features = self.low_level_conv(low_level_features)
        x = F.interpolate(x, size=low_level_features.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, low_level_features], dim=1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.upsample(x)
        
        return x


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


def calculate_f1_score(pred, target):
    # Calculate F1 score
    f1 = f1_score(target.flatten().cpu(),
                  pred.flatten().cpu(), average='macro')

    return f1