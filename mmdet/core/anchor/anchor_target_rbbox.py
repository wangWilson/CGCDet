import torch

from ..bbox import assign_and_sample, build_assigner, \
    PseudoSampler, bbox2delta, dbbox2delta, dbbox2delta_v3, hbb2obb_v2
from ..utils import multi_apply
from mmdet.core.bbox.transforms_rbbox import gt_mask_bp_obbs


def anchor_target_rbbox(anchor_list,
                  guided_anchors_list_R,
                  valid_flag_list,
                  gt_bboxes_list,
                  gt_masks_list,
                  img_metas,
                  target_means,
                  target_stds,
                  cfg,
                  gt_bboxes_ignore_list=None,
                  gt_labels_list=None,
                  label_channels=1,
                  sampling=True,
                  unmap_outputs=True,
                  with_module=False,
                  hbb_trans='hbb2obb_v2'):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        guided_anchors_list_R[i] = torch.cat(guided_anchors_list_R[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list) = multi_apply(
         anchor_target_rbbox_single,
         anchor_list,
         guided_anchors_list_R,
         valid_flag_list,
         gt_bboxes_list,
         gt_masks_list,
         gt_bboxes_ignore_list,
         gt_labels_list,
         img_metas,
         target_means=target_means,
         target_stds=target_stds,
         cfg=cfg,
         label_channels=label_channels,
         sampling=sampling,
         unmap_outputs=unmap_outputs,
         with_module=with_module,
         hbb_trans=hbb_trans)
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_total_pos, num_total_neg)


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets

def obbxyxy2xywh(boxes):
    """
    fix a bug
    :param boxes: shape (n, 4) (xmin, ymin, xmax, ymax)
    :return: dbboxes: shape (n, 5) (x_ctr, y_ctr, w, h, angle)
    """
    num_boxes = boxes.size(0)
    ex_heights = boxes[..., 2] - boxes[..., 0] + 1.0
    ex_widths = boxes[..., 3] - boxes[..., 1] + 1.0
    ex_ctr_x = boxes[..., 0] + 0.5 * (ex_heights - 1.0)
    ex_ctr_y = boxes[..., 1] + 0.5 * (ex_widths - 1.0)
    c_bboxes = torch.cat((ex_ctr_x.unsqueeze(1), ex_ctr_y.unsqueeze(1), ex_widths.unsqueeze(1), ex_heights.unsqueeze(1)), 1)
    initial_angles = boxes[..., 4].unsqueeze(1)#-c_bboxes.new_ones((num_boxes, 1)) * np.pi / 2
    # initial_angles = -torch.ones((num_boxes, 1)) * np.pi/2
    dbboxes = torch.cat((c_bboxes, initial_angles), 1)

    return dbboxes

def anchor_target_rbbox_single(flat_anchors,
                         guided_anchors_R,
                         valid_flags,
                         gt_bboxes,
                         gt_masks,
                         gt_bboxes_ignore,
                         gt_labels,
                         img_meta,
                         target_means,
                         target_stds,
                         cfg,
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True,
                         with_module=False,
                         hbb_trans='hbb2obb_v2'):
    # print(flat_anchors.shape)
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors
    # import pdb
    # print('in anchor target rbbox single')
    # pdb.set_trace()
    # print(inside_flags.shape)
    # print(flat_anchors.shape)
    anchors = flat_anchors[inside_flags, :]
    anchors_R = guided_anchors_R[inside_flags, :]

    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)

    num_valid_anchors = anchors.shape[0]

    bbox_targets =  torch.zeros(num_valid_anchors, 5).to(anchors.device)
    bbox_weights = torch.zeros(num_valid_anchors, 5).to(anchors.device)



    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds

    # TODO: copy the code in mask target to here. trans gt_masks to gt_rbboxes
    pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
    # implementation A
    # pos_gt_masks = gt_masks[pos_assigned_gt_inds.cpu().numpy()]
    # pos_gt_obbs = gt_mask_bp_obbs(pos_gt_masks)
    # pos_gt_obbs_ts = torch.from_numpy(pos_gt_obbs).to(sampling_result.pos_bboxes.device)
    # implementation B
    gt_obbs = gt_mask_bp_obbs(gt_masks, with_module)
    gt_obbs_ts = torch.from_numpy(gt_obbs).to(sampling_result.pos_bboxes.device)
    pos_gt_obbs_ts = gt_obbs_ts[pos_assigned_gt_inds]

    # print('with_module',with_module)

    if len(pos_inds) > 0:

        ################## use pred rotate anchor
        # pos_ext_bboxes_0 = hbb2obb_v2(sampling_result.pos_bboxes)
        # pos_ext_bboxes = anchors_R[sampling_result.pos_inds]     
        pos_ext_bboxes = obbxyxy2xywh(anchors_R[sampling_result.pos_inds])   

        # pos_ext_bboxes[:,:4]=pos_ext_bboxes_0[:,:4]  
        if with_module:
            pos_bbox_targets = dbbox2delta(pos_ext_bboxes,
                                           pos_gt_obbs_ts,
                                           target_means, target_stds)
        else:
            pos_bbox_targets = dbbox2delta_v3(pos_ext_bboxes,
                                              pos_gt_obbs_ts,
                                              target_means, target_stds)
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
    

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds)


def anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                        allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border) & \
            (flat_anchors[:, 1] >= -allowed_border) & \
            (flat_anchors[:, 2] < img_w + allowed_border) & \
            (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
