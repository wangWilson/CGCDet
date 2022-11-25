import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn

from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule
from mmdet.core import mask_target,multi_apply



@HEADS.register_module
class FCNMaskHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method=None,
                 # upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 class_agnostic=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(FCNMaskHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size  # WARN: not used and reserved
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.loss_mask = build_loss(loss_mask)

        self.convs0 = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs0.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

        self.convs1 = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs1.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

        self.convs2 = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs2.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

        self.convs3 = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs3.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

        self.convs4 = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs4.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))



        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        if self.upsample_method is None:
            self.upsample0 = None
            self.upsample1 = None
            self.upsample2 = None
            self.upsample3 = None
            self.upsample4 = None

        elif self.upsample_method == 'deconv':
            self.upsample0 = nn.ConvTranspose2d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
            self.upsample1 = nn.ConvTranspose2d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
            self.upsample2 = nn.ConvTranspose2d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
            self.upsample3 = nn.ConvTranspose2d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
            self.upsample4 = nn.ConvTranspose2d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)


        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)

        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits0 = nn.Conv2d(logits_in_channel, out_channels, 1)
        self.conv_logits1 = nn.Conv2d(logits_in_channel, out_channels, 1)
        self.conv_logits2 = nn.Conv2d(logits_in_channel, out_channels, 1)
        self.conv_logits3 = nn.Conv2d(logits_in_channel, out_channels, 1)
        self.conv_logits4 = nn.Conv2d(logits_in_channel, out_channels, 1)


        self.relu0 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)

        self.debug_imgs = None

    def init_weights(self):
        for m in [self.upsample0, self.conv_logits0]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        for m in [self.upsample1, self.conv_logits1]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        for m in [self.upsample2, self.conv_logits2]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        for m in [self.upsample3, self.conv_logits3]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        for m in [self.upsample4, self.conv_logits4]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)


    # def forward(self, x):
    #     return multi_apply(self.forward_single, x)

    # def forward_single(self, x):
    def forward(self, x):      
        # print(len(x))
        # for x_i in x:
        #    print('x_i',x_i.shape) 
        x0=x[0]
        x1=x[1]
        x2=x[2]
        x3=x[3]
        x4=x[4]
        for conv in self.convs0:
            x0 = conv(x0)
        if self.upsample0 is not None:
            x0 = self.upsample0(x0)
            if self.upsample_method == 'deconv':
                x0 = self.relu0(x0)
        mask_pred0 = self.conv_logits0(x0)

        for conv in self.convs1:
            x1 = conv(x1)
        if self.upsample1 is not None:
            x1 = self.upsample1(x1)
            if self.upsample_method == 'deconv':
                x1 = self.relu1(x1)
        mask_pred1 = self.conv_logits1(x1)

        for conv in self.convs2:
            x2 = conv(x2)
        if self.upsample2 is not None:
            x2 = self.upsample2(x2)
            if self.upsample_method == 'deconv':
                x2 = self.relu2(x2)
        mask_pred2 = self.conv_logits2(x2)

        for conv in self.convs3:
            x3 = conv(x3)
        if self.upsample3 is not None:
            x3 = self.upsample3(x3)
            if self.upsample_method == 'deconv':
                x3 = self.relu3(x3)
        mask_pred3 = self.conv_logits3(x3)

        for conv in self.convs4:
            x4 = conv(x4)
        if self.upsample4 is not None:
            x4 = self.upsample4(x4)
            if self.upsample_method == 'deconv':
                x4 = self.relu0(x4)
        mask_pred4 = self.conv_logits0(x4)
        mask_pred=tuple(map(list, zip((mask_pred0,mask_pred1,mask_pred2,mask_pred3,mask_pred4))))
        # for mask in mask_pred:
        #     for i,mask_i in enumerate(mask):
        #        print('mask',i,mask_i.shape)
        return mask_pred

    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    def loss1(self, mask_pred, mask_targets):
        loss = dict()
        if self.class_agnostic:#True
            # loss_mask = self.loss_mask(mask_pred, mask_targets,
            #                            torch.zeros_like(labels))
            loss_mask = self.loss_mask(mask_pred, mask_targets)

        loss['loss_mask'] = loss_mask
        return loss

    def single_loss(self, mask_pred, mask_targets):       
        loss_mask_all=[]
        # for i in range(len(mask_pred)):

        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets)

           # loss_mask_all.append(loss_mask)

        return loss_mask

    def loss(self, mask_pred, mask_targets):
        loss_list=[]
        for i in range(5):
           # print('mask_pred',mask_pred[i][0][0].shape)
           loss=self.single_loss(mask_pred[i][0][0], mask_targets[0][i])
           loss1=self.single_loss(mask_pred[i][0][1], mask_targets[1][i])
           # loss= multi_apply(self.single_loss,mask_pred, mask_targets)
           loss_list.append(loss)
           loss_list.append(loss1)
        loss_mask=dict()

        loss_mask['loss_mask']=loss_list
        return loss_mask



    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)

        cls_segms = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            if not self.class_agnostic:
                mask_pred_ = mask_pred[i, label, :, :]
            else:
                mask_pred_ = mask_pred[i, 0, :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label - 1].append(rle)

        return cls_segms
