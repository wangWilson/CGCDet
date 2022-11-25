# Getting Started

This page provides basic tutorials about the usage of mmdetection.
For installation instructions, please see [INSTALL.md](doc/INSTALL.md).



## Inference with pretrained models

## Pretrained models

Sorry, because the size of our pretrained model exceeds the supplementary Material size limit, we can't upload the model temporarily.

### Test a dataset

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}]
```

Optional arguments:
- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.

Examples:

Assume that you have already downloaded the checkpoints to `work_dirs/`.

1. Test Faster R-CNN with RT + CGC loss + OCP-Guided Lable assignment in single-scale training and test

```shell
python tools/test.py configs/CGC/faster_rcnn_RoITrans_r50_fpn_1x_dota_CGC_OCP_single_data.py \
    work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota_CGC_OCP_single_data/epoch_12.pth \ 
    --out work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota_CGC_OCP_single_data/results.pkl
```

2. Test Faster R-CNN with RT + CGC loss + OCP-Guided Lable assignment in multi-scale training and test

```shell
./tools/dist_test.sh configs/CGC/faster_rcnn_RoITrans_r50_fpn_1x_dota_CGC_OCP_single_data.py \
    work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota_CGC_OCP_mutlscale_data/epoch_12.pth \
    4 --out work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota_CGC_OCP_mutlscale_data/results.pkl 
```



## Train a model

mmdetection implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

**\*Important\***: The default learning rate in config files is for 8 GPUs.
If you use less or more than 8 GPUs, you need to set the learning rate proportional
to the GPU num, e.g., 0.01 for 4 GPUs and 0.04 for 16 GPUs.

### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE}
```

If you want to specify the working directory in the command, you can add an argument `--work_dir ${YOUR_WORK_DIR}`.

### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--validate` (recommended): Perform evaluation at every k (default=1) epochs during the training.
- `--work_dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.

### Train with multiple machines

If you run mmdetection on a cluster managed with [slurm](https://slurm.schedmd.com/), you can just use the script `slurm_train.sh`.

```shell
./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} [${GPUS}]
```




