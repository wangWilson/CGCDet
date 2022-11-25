from .two_stage_rbbox import TwoStageDetectorRbbox
from ..registry import DETECTORS


@DETECTORS.register_module

class FasterRCNNOBB(TwoStageDetectorRbbox):

    def __init__(self,
                 backbone,
                 rpn_head,

                 rbbox_roi_extractor,
                 rbbox_head,
                 train_cfg,
                 test_cfg,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 neck=None,
                 shared_head=None,
                 shared_head_rbbox=None,
                 pretrained=None):
        super(FasterRCNNOBB, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            shared_head_rbbox=shared_head_rbbox,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            rbbox_roi_extractor=rbbox_roi_extractor,
            rbbox_head=rbbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
