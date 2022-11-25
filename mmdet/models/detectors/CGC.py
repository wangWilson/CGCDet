from __future__ import division

import torch
import torch.nn as nn

from .base_new import BaseDetectorNew
from .test_mixins import RPNTestMixin
from .. import builder
from ..registry import DETECTORS
from ..builder import build_loss
from mmdet.core import (build_assigner, bbox2roi, dbbox2roi, bbox2result, build_sampler, 
                        dbbox2result, merge_aug_masks, roi2droi, mask2poly,
                        get_best_begin_point, polygonToRotRectangle_batch,RotBox2Polys_torch,
                        gt_mask_bp_obbs_list, choose_best_match_batch,   delta2dbbox_v3,
                        choose_best_Rroi_batch, dbbox_rotate_mapping, bbox_rotate_mapping)
from mmdet.core import (bbox_mapping, merge_aug_proposals, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms, merge_rotate_aug_proposals,
                        merge_rotate_aug_bboxes, multiclass_nms_rbbox)
import copy
from mmdet.core import RotBox2Polys, polygonToRotRectangle_batch

import torch




@DETECTORS.register_module
class CGC(BaseDetectorNew, RPNTestMixin):

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
        assert bbox_roi_extractor is not None
        assert bbox_head is not None

        assert rbbox_roi_extractor is not None
        assert rbbox_head is not None
        super(RoITransformer, self).__init__()

        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if shared_head_rbbox is not None:
            self.shared_head_rbbox = builder.build_shared_head(shared_head_rbbox)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)
        # import pdb
        # pdb.set_trace()
        if rbbox_head is not None:
            self.rbbox_roi_extractor = builder.build_roi_extractor(
                rbbox_roi_extractor)
            # print(rbbox_head)
            self.rbbox_head = builder.build_head(rbbox_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.rbbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        loss_shape=dict(type='IoULoss', style='bounded', beta=0.2, loss_weight=1.0)


        self.loss_shape = build_loss(loss_shape)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(RoITransformer, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_shared_head_rbbox:
            self.shared_head_rbbox.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_rbbox:
            self.rbbox_roi_extractor.init_weights()
            self.rbbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def CGC_loss(self,rois,rois_gt,bbox_preds,labels,bbox_weights,img_metas):
    
        bboxes_list = []
        bboxes_gt_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i).squeeze()
            num_rois = inds.numel()
            
            bboxes_ = rois[inds, 1:]#(x_cetr,x_cetr,w,h,theta)
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            rois_gt_ = rois_gt[inds, 1:]#x1,y1,x2,y2

    
            target_means=[0., 0., 0., 0., 0.]
            target_stds=[0.1, 0.1, 0.2, 0.2, 0.1]
            new_rois = delta2dbbox_v3(bboxes_, bbox_pred_, target_means,
                              target_stds, img_meta_['img_shape'])##(512,5)(x_cetr,x_cetr,w,h,theta)
           cs = torch.cos(new_rois[:, 4])
            ss = torch.sin(new_rois[:, 4])

            inds = cs<0
            cs[inds] = -1 * cs[inds]
            inds = ss<0
            ss[inds] = -1 * ss[inds]

            new_w = new_rois[:, 2]*cs+new_rois[:, 3]*ss
            new_h = new_rois[:, 2]*ss+new_rois[:, 3]*cs

            xmin = new_rois[:,0] - (new_w-1) / 2.0
            ymin = new_rois[:,1] - (new_h-1) / 2.0
            xmax = new_rois[:,0] + (new_w-1) / 2.0
            ymax = new_rois[:,1] + (new_h-1) / 2.0

            new_rois_4=torch.cat((xmin.unsqueeze(1), ymin.unsqueeze(1), 
                                  xmax.unsqueeze(1), ymax.unsqueeze(1)), 1)
    

            bboxes_list.append(new_rois_4)
            bboxes_gt_list.append(rois_gt_)

        bboxes=torch.cat((bboxes_list),0)#(1024,4)
        bboxes_gt=torch.cat((bboxes_gt_list),0)

        pos_inds = labels > 0

        loss_shape = self.loss_shape(bboxes[pos_inds],
                                     bboxes_gt[pos_inds],
                                     bbox_weights[pos_inds],
                                     avg_factor=bboxes_gt.size(0))
        return dict(loss_consis=loss_shape)


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

        # trans gt_masks to gt_obbs
        gt_obbs = gt_mask_bp_obbs_list(gt_masks)

        # RPN forward and loss
        if self.with_rpn:
            # print(111)

            if gt_bboxes[0].shape[0]>300:
                gt_bboxes[0]=gt_bboxes[0][:200,:]
                gt_masks[0]=gt_masks[0][:200,:,:]
                gt_labels[0]=gt_labels[0][:200]
                gt_obbs[0]=gt_obbs[0][:200,:]

            rpn_outs = self.rpn_head(x)

            rpn_loss_inputs = rpn_outs + (gt_bboxes,gt_masks, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)

            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals (hbb assign)
        if self.with_bbox or self.with_mask:
            # print(222)
            bbox_assigner = build_assigner(self.train_cfg.rcnn[0].assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn[0].sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])

            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)

            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            ## rbbox
            rbbox_targets = self.bbox_head.get_target(
                sampling_results, gt_masks, gt_labels, self.train_cfg.rcnn[0])
            # print('cls_score',cls_score.shape,bbox_pred.shape)
            labels_consis,label_weights_consis,bbox_targets_consis,bbox_weights_consis=rbbox_targets
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *rbbox_targets)
            # losses.update(loss_bbox)
            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(0, name)] = (value)

            loss_bbox_CGC = self.CGC_loss(roi2droi(rois),rois,bbox_pred,labels_consis,bbox_weights_consis,img_meta)
            losses.update(loss_bbox_CGC)


        pos_is_gts = [res.pos_is_gt for res in sampling_results]
        roi_labels = rbbox_targets[0]
        with torch.no_grad():
            # import pdb
            # pdb.set_trace()
            rotated_proposal_list = self.bbox_head.refine_rbboxes(
                roi2droi(rois), roi_labels, bbox_pred, pos_is_gts, img_meta
            )
        # import pdb
        # pdb.set_trace()
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
            # feat enlarge

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
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']

        rcnn_test_cfg = self.test_cfg.rcnn

        rois = bbox2roi(proposal_list)
        bbox_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_label = cls_score.argmax(dim=1)
        rrois = self.bbox_head.regress_by_class_rbbox(roi2droi(rois), bbox_label, bbox_pred,
                                                      img_meta[0])

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
            cfg=rcnn_test_cfg)
        rbbox_results = dbbox2result(det_rbboxes, det_labels,
                                     self.rbbox_head.num_classes)

        return rbbox_results

    def aug_test(self, imgs, img_metas, proposals=None, rescale=None):
        # raise NotImplementedError
        # import pdb; pdb.set_trace()
        proposal_list = self.aug_test_rpn_rotate(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)

        rcnn_test_cfg = self.test_cfg.rcnn

        aug_rbboxes = []
        aug_rscores = []
        for x, img_meta in zip(self.extract_feats(imgs), img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)

            angle = img_meta[0]['angle']
            # print('img shape: ', img_shape)
            if angle != 0:
                try:

                    proposals = bbox_rotate_mapping(proposal_list[0][:, :4], img_shape,
                                                angle)
                except:
                    import pdb; pdb.set_trace()
            rois = bbox2roi([proposals])
            # recompute feature maps to save GPU memory
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            cls_score, bbox_pred = self.bbox_head(roi_feats)


            bbox_label = cls_score.argmax(dim=1)
            rrois = self.bbox_head.regress_by_class_rbbox(roi2droi(rois), bbox_label,
                                                          bbox_pred,
                                                          img_meta[0])

            rrois_enlarge = copy.deepcopy(rrois)
            rrois_enlarge[:, 3] = rrois_enlarge[:, 3] * self.rbbox_roi_extractor.w_enlarge
            rrois_enlarge[:, 4] = rrois_enlarge[:, 4] * self.rbbox_roi_extractor.h_enlarge
            rbbox_feats = self.rbbox_roi_extractor(
                x[:len(self.rbbox_roi_extractor.featmap_strides)], rrois_enlarge)
            if self.with_shared_head_rbbox:
                rbbox_feats = self.shared_head_rbbox(rbbox_feats)

            rcls_score, rbbox_pred = self.rbbox_head(rbbox_feats)
            rbboxes, rscores = self.rbbox_head.get_det_rbboxes(
                rrois,
                rcls_score,
                rbbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=None)
            aug_rbboxes.append(rbboxes)
            aug_rscores.append(rscores)

        merged_rbboxes, merged_rscores = merge_rotate_aug_bboxes(
            aug_rbboxes, aug_rscores, img_metas, rcnn_test_cfg
        )
        det_rbboxes, det_rlabels = multiclass_nms_rbbox(
                                merged_rbboxes, merged_rscores, rcnn_test_cfg.score_thr,
                                rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)

        if rescale:
            _det_rbboxes = det_rbboxes
        else:
            _det_rbboxes = det_rbboxes.clone()
            _det_rbboxes[:, :4] *= img_metas[0][0]['scale_factor']

        rbbox_results = dbbox2result(_det_rbboxes, det_rlabels,
                                     self.rbbox_head.num_classes)
        return rbbox_results









