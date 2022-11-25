from __future__ import division
import torch
import torch.nn as nn

from .base import BaseDetector
from .base_new import BaseDetectorNew
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin

from .. import builder
from ..registry import DETECTORS


from mmdet.core import (build_assigner, bbox2roi, dbbox2roi, bbox2result, build_sampler,
                        dbbox2result, merge_aug_masks, roi2droi, mask2poly,
                        get_best_begin_point, polygonToRotRectangle_batch,
                        gt_mask_bp_obbs_list, choose_best_match_batch,
                        choose_best_Rroi_batch, dbbox_rotate_mapping, bbox_rotate_mapping)



import numpy as np
import cv2
from ..mask_heads import FCNMaskHead
import copy


@DETECTORS.register_module
class TwoStageDetectorRbbox(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 shared_head_rbbox=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 rbbox_roi_extractor=None,
                 rbbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        # assert bbox_roi_extractor is not None
        # assert bbox_head is not None

        assert rbbox_roi_extractor is not None
        assert rbbox_head is not None

        super(TwoStageDetectorRbbox, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if rbbox_head is not None:
            self.rbbox_roi_extractor = builder.build_roi_extractor(
                rbbox_roi_extractor)

            self.rbbox_head = builder.build_head(rbbox_head)


        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        self.angle_attention_mask=FCNMaskHead()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.k=0

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetectorRbbox, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()

        if self.with_shared_head_rbbox:
            self.shared_head_rbbox.init_weights(pretrained=pretrained)
        if self.with_rbbox:
            self.rbbox_roi_extractor.init_weights()
            self.rbbox_head.init_weights()

        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()
        self.angle_attention_mask.init_weights()



    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    def forward_convert(self,rect):

        box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
        box = np.reshape(box, [-1, ])
        # boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]])
        boxes=[[box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]]]

        return np.array(boxes, dtype=np.float32)

    def get_mask(self,imgs, boxes):
       mask_targets=[]
       # print('imgs.shape',imgs.shape)
       # print('boxes.shape',boxes[0].shape)
       # print('boxes.shape',boxes[1].shape)
       mask_list=[]
       for i,img in enumerate(imgs):
          _,h, w = img.shape
          mask = np.zeros([h, w])
          for b in boxes[i]:

              b=self.forward_convert(b)
              # b = np.reshape(b[0:-1], [4, 2])
              b = np.reshape(b, [4, 2])
              rect = np.array(b, np.int32)
              cv2.fillConvexPoly(mask, rect, 1)
          # mask = cv2.resize(mask, dsize=(h // 16, w // 16))
          # mask = np.expand_dims(mask, axis=-1)
          mask_list.append(mask)
          # self.k+=1

       return np.array(mask_list, np.float32)


    def target_mask_to_tensor(self,mask,x):

        mask=torch.from_numpy(mask).float().to(x[0][0].device)
        mask_list=[]
        for i in range(5):
            mask_i=mask.resize_(int(256/2**i),int(256/2**i))
            # mask_i=mask.resize_(int(128/2**i),int(128/2**i))

            mask_i=mask_i.unsqueeze(dim=0)
            # mask_i=mask_i.unsqueeze(dim=0)

            mask_list.append(mask_i)
        return mask_list

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        x = self.extract_feat(img)

        losses = dict()
        
        gt_obbs = gt_mask_bp_obbs_list(gt_masks)


        # RPN forward and loss
        if self.with_rpn:
            ##RPN 预测 前景背景分类 和bbox偏置
            rpn_outs = self.rpn_head(x)
            # rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
            #                               self.train_cfg.rpn)

            # rpn_losses = self.rpn_head.loss(
            #     *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            # rpn_loss_inputs = rpn_outs + (gt_bboxes,gt_masks,gt_labels, img_meta,
            #                               self.train_cfg.rpn)
            # print()
            # rpn_losses = self.rpn_head.loss(
            #     *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

            # if gt_bboxes[0].shape[0]>300:
            #     gt_bboxes[0]=gt_bboxes[0][:250,:]
            #     gt_masks[0]=gt_masks[0][:250,:,:]
            #     gt_labels[0]=gt_labels[0][:250]
            #     gt_obbs[0]=gt_obbs[0][:250,:]
            #Ga RPN
            rpn_loss_inputs = rpn_outs + (gt_bboxes,gt_masks,gt_labels, img_meta,
                                          self.train_cfg.rpn)
            # print()
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)


            losses.update(rpn_losses)
            ###从RPN预测的偏执转成proposal
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            # print(proposal_cfg)
            # print(self.test_cfg.rpn)
            # exit(1)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)


        # print(proposal_list[0][-1])
        rotated_proposal_list=proposal_list

        # assign gts and sample proposals (rbb assign)
        if self.with_rbbox:
            # print(444)
            bbox_assigner = build_assigner(self.train_cfg.rcnn[1].assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn[1].sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []

            for i in range(num_imgs):
                gt_obbs_best_roi = choose_best_Rroi_batch(gt_obbs[i])
                # print(len(rotated_proposal_list))
                # for ii in range(len(rotated_proposal_list)):
                #    for j,bbox in enumerate(rotated_proposal_list[ii]):
                #       print('bbox.shape',ii,j,rotated_proposal_list.shape)

                assign_result = bbox_assigner.assign(
                    rotated_proposal_list[i], gt_obbs_best_roi, gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    rotated_proposal_list[i],
                    torch.from_numpy(gt_obbs_best_roi).float().to(rotated_proposal_list[i].device),
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        if self.with_rbbox:
            # print(555)
            # (batch_ind, x_ctr, y_ctr, w, h, angle)
            rrois = dbbox2roi([res.bboxes for res in sampling_results])

            rrois[:, 3] = rrois[:, 3] * self.rbbox_roi_extractor.w_enlarge
            rrois[:, 4] = rrois[:, 4] * self.rbbox_roi_extractor.h_enlarge
            rbbox_feats = self.rbbox_roi_extractor(x[:self.rbbox_roi_extractor.num_inputs],
                                                   rrois)
            

            if self.with_shared_head_rbbox:
                rbbox_feats = self.shared_head_rbbox(rbbox_feats)
            cls_score, rbbox_pred = self.rbbox_head(rbbox_feats)
            rbbox_targets = self.rbbox_head.get_target_rbbox(sampling_results, gt_obbs,
                                                        gt_labels, self.train_cfg.rcnn[1])
            loss_rbbox = self.rbbox_head.loss(cls_score, rbbox_pred, *rbbox_targets)
            for name, value in loss_rbbox.items():
                losses['s{}.{}'.format(1, name)] = (value)

        return losses



    def simple_test(self, img, img_meta,scale_factor, proposals=None, rescale=False):
        """Test without augmentation."""
        # assert self.with_bbox, "Bbox head must be implemented."
        x = self.extract_feat(img)
       
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
        # print('test scale_factor',scale_factor)
        img_shape = img_meta[0]['img_shape']
        rescale=False
        # scale_factor = img_meta[0]['scale_factor']
        scale_factor = torch.tensor(float(scale_factor))
        # for proposals in proposal_list:
        #     for proposal in proposals:
        #        print(proposal.shape)

        # det_bboxes, det_labels = self.simple_test_bboxes(
        #     x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)

        # bbox_results = dbbox2result(det_bboxes, det_labels,
        #                            self.bbox_head.num_classes)
        # print(img_meta)
        rrois = dbbox2roi(proposal_list)
        rrois_enlarge = copy.deepcopy(rrois)
        rrois_enlarge[:, 3] = rrois_enlarge[:, 3] * self.rbbox_roi_extractor.w_enlarge
        rrois_enlarge[:, 4] = rrois_enlarge[:, 4] * self.rbbox_roi_extractor.h_enlarge
        rbbox_feats = self.rbbox_roi_extractor(
            x[:len(self.rbbox_roi_extractor.featmap_strides)], rrois_enlarge)
        if self.with_shared_head_rbbox:
            rbbox_feats = self.shared_head_rbbox(rbbox_feats)

        rcls_score, rbbox_pred = self.rbbox_head(rbbox_feats)
        det_rbboxes, det_labels = self.rbbox_head.get_det_rbboxes(
            rrois,
            rcls_score,
            rbbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=self.test_cfg.rcnn)
        rbbox_results = dbbox2result(det_rbboxes, det_labels,
                                     self.rbbox_head.num_classes)


        if not self.with_mask:
            return rbbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        # print(img_metas)
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)

        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)


        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results
