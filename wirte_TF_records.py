from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import copy


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# write sample to tfrecords
# require:
# 1.complete sample
# 2.tfrecords' writer with 48 scale
# 3.tfrecords' writer with 24 scale
# 4.tfrecords' writer with 12 scale
# 5.The old count of sample
# return:
# 1.The new count of sample
def from_rect_sample_import_tfrecords(rect_sample_scale, writer_48, writer_24, writer_12, num):
    for sample_scale in rect_sample_scale:
        im = sample_scale[0]
        for diff_GT in sample_scale[1]:
            for sample_label in range(3):
                # sample set
                label = 1 if sample_label == 0 else 0
                for rects_pos in diff_GT[sample_label]:
                    scale = rects_pos[2]
                    roi_im = copy.deepcopy(im[rects_pos[1]:rects_pos[1] + scale, rects_pos[0]:rects_pos[0] + scale])
                    rect_a = np.array(rects_pos)
                    rect_GT = np.array(diff_GT[3])
                    # save
                    im_raw = roi_im.astype(np.uint8).tostring()
                    rect_a_raw = rect_a.astype(np.int32).tostring()
                    rect_GT_raw = rect_GT.astype(np.int32).tostring()
                    if sample_label == 0 and scale == 48:
                        num += 1
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'label': _int64_feature(label),
                        'im_raw': _bytes_feature(im_raw),
                        'rect_a_raw': _bytes_feature(rect_a_raw),
                        'rect_GT_raw': _bytes_feature(rect_GT_raw)}))
                    if scale == 48:
                        writer_48[sample_label].write(example.SerializeToString())
                    elif scale == 24:
                        writer_24[sample_label].write(example.SerializeToString())
                    elif scale == 12:
                        writer_12[sample_label].write(example.SerializeToString())
    return num
