import BboxToolkit as bt
import os
import os.path as osp
import argparse
import numpy as np

from random import shuffle
from multiprocessing import Pool
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser(description='visualization')

    # arguments for loading data
    parser.add_argument('--load_type', type=str, help='dataset and save form')
    parser.add_argument('--img_dir', type=str, help='path to images')
    parser.add_argument('--ann_dir', type=str, default=None,
                        help='path to annotations')
    parser.add_argument('--classes', nargs='+', type=str, default=None,
                        help='the classes to visualize')
    parser.add_argument('--nproc', type=int, default=10,
                        help='the procession number for loading data')

    # arguments for selecting content
    parser.add_argument('--skip_empty', action='store_true',
                        help='whether show images without objects')
    parser.add_argument('--random_vis', action='store_true',
                        help='whether to shuffle the order of images')
    parser.add_argument('--ids', nargs='+', type=str, default=None,
                        help='choice id to visualize')
    parser.add_argument('--show_off', action='store_true',
                        help='stop showing images')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='whether to save images and where to save images')
    parser.add_argument('--vis_nproc', type=int, default=10,
                        help='the procession number for visualizing')

    # arguments for visualisation
    parser.add_argument('--score_thr', type=float, default=0.2,
                        help='the score threshold for bboxes')
    parser.add_argument('--thickness', type=int, default=2,
                        help='the thickness for bboxes')
    parser.add_argument('--font_scale', type=float, default=1,
                        help='the thickness for font')
    parser.add_argument('--wait_time', type=int, default=0,
                        help='wait time for showing images')
    parser.add_argument('--max_size', type=int, default=1000,
                        help='wait time for showing images')
    args = parser.parse_args()
    assert args.load_type is not None, "argument load_type can't be None"
    assert args.img_dir is not None, "argument img_dir can't be None"
    assert args.save_dir or (not args.show_off)

    return args


def single_vis(content, ids, img_dir, save_dir, class_names, score_thr,
               thickness, font_scale, show_off, wait_time, max_size):
    if ids is not None and content['id'] not in ids:
        return

    imgpath = osp.join(img_dir, content['filename'])
    out_file = osp.join(save_dir, content['filename']) \
            if save_dir else None
    if 'ann' in content:
        ann = content['ann']
        bboxes = ann['bboxes']
        labels = ann['labels']
        scores = ann.get('scores', None)
    else:
        bboxes = np.zeros((0, 4), dtype=np.float)
        labels = np.zeros((0, ), dtype=np.int)
        scores = None

    print(imgpath)
    bt.imshow_det_bboxes(imgpath, bboxes, labels,
                         scores=scores,
                         class_names=class_names,
                         score_thr=score_thr,
                         thickness=thickness,
                         font_scale=font_scale,
                         show=(not show_off),
                         wait_time=wait_time,
                         max_size=max_size,
                         out_file=out_file)


def main():
    args = parse_args()

    print(f'{args.load_type} loading!')
    load_func = getattr(bt.datasets, 'load_'+args.load_type)
    contents, classes = load_func(
        img_dir=args.img_dir,
        ann_dir=args.ann_dir,
        classes=args.classes,
        nproc=args.nproc)

    if args.skip_empty:
        contents = [content for content in contents
                    if content['ann']['bboxes'].size > 0]
    if args.random_vis:
        shuffle(contents)
    if args.save_dir and (not osp.exists(args.save_dir)):
        os.makedirs(args.save_dir)

    _vis_func = partial(single_vis,
                        ids=args.ids,
                        img_dir=args.img_dir,
                        save_dir=args.save_dir,
                        class_names=classes,
                        score_thr=args.score_thr,
                        thickness=args.thickness,
                        font_scale=args.font_scale,
                        show_off=args.show_off,
                        wait_time=args.wait_time,
                        max_size=args.max_size)
    if args.show_off and args.vis_nproc > 1:
        pool = Pool(args.vis_nproc)
        pool.map(_vis_func, contents)
        pool.close()
    else:
        list(map(_vis_func, contents))


if __name__ == '__main__':
    main()
