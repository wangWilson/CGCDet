# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import math
import sys
import os

from lib import cfgs


def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_feature_map_size(src_len):
    feature_map_size = []
    src_len /= 2 ** (int(cfgs.LEVEL[0][-1])-1)
    for _ in range(len(cfgs.LEVEL)):
        src_len = math.ceil(src_len / 2)
        feature_map_size.append((src_len, src_len))

    return feature_map_size


def get_dota_short_names(label):
    DOTA_SHORT_NAMES = {
        'roundabout': 'RA',
        'tennis-court': 'TC',
        'swimming-pool': 'SP',
        'storage-tank': 'ST',
        'soccer-ball-field': 'SBF',
        'small-vehicle': 'SV',
        'ship': 'SH',
        'plane': 'PL',
        'large-vehicle': 'LV',
        'helicopter': 'HC',
        'harbor': 'HA',
        'ground-track-field': 'GTF',
        'bridge': 'BR',
        'basketball-court': 'BC',
        'baseball-diamond': 'BD'
    }

    return DOTA_SHORT_NAMES[label]