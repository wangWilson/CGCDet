import mmcv

from . import assigners, samplers


def build_assigner(cfg, **kwargs):
    if isinstance(cfg, assigners.BaseAssigner):
        return cfg
    elif isinstance(cfg, dict):
        return mmcv.runner.obj_from_dict(cfg, assigners, default_args=kwargs)
    else:
        raise TypeError('Invalid type {} for building a sampler'.format(
            type(cfg)))


def build_sampler(cfg, **kwargs):
    if isinstance(cfg, samplers.BaseSampler):
        return cfg
    elif isinstance(cfg, dict):
        return mmcv.runner.obj_from_dict(cfg, samplers, default_args=kwargs)
    else:
        raise TypeError('Invalid type {} for building a sampler'.format(
            type(cfg)))


def assign_and_sample(bboxes, gt_bboxes, gt_bboxes_ignore, gt_labels, cfg,gt_obbs=None):
    bbox_assigner = build_assigner(cfg.assigner)
    bbox_sampler = build_sampler(cfg.sampler)
    # print('gt_obbs',gt_obbs)
    if gt_obbs.any()!=None:
       # print(1111)
       assign_result = bbox_assigner.assign(bboxes, gt_bboxes, gt_bboxes_ignore,
                                         gt_labels,gt_obbs)
    else:
       # print(2222)
       assign_result = bbox_assigner.assign(bboxes, gt_bboxes, gt_bboxes_ignore,
                                         gt_labels) 

    # assign_result = bbox_assigner.assign(bboxes, gt_bboxes,gt_obbs, gt_bboxes_ignore,
    #                                      gt_labels)
    sampling_result = bbox_sampler.sample(assign_result, bboxes, gt_bboxes,
                                          gt_labels)
    return assign_result, sampling_result
