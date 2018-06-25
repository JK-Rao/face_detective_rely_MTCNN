import numpy as np
import cv2
from os.path import join
import time
import copy
import tensorflow as tf
import wirte_TF_records

PATH = '/home/jr/Downloads'


# read annotation file
# require:
# return:
# 1.The list of annotation file's text lines
def from_file_import_datalist(set_type):
    filePath = join(PATH, 'wider_face_split/wider_face_%s_bbx_gt.txt' % set_type)
    with open(filePath, 'r') as f:
        fileLines = f.readlines()
    return fileLines


# read images' absolute path and it's ground truth base on fileLines
# data format is:
# The format of txt ground truth.
# File name
# Number of bounding box
# x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
# require:
# 1.The list of annotation file's text lines
# return:
# 1.img_paths: List include img absolute path with string format
# 2.img_GTs: like [[(rect1 of img1),(rect2 of img1)...],[(rect1 of img2),(rect2 of img2)...]...]
def from_datalist_import_im_GT(fileLines):
    pass_indicate = 0
    line_function = 'im_path'
    new_im_sample = True
    img_paths = []
    img_GTs = []
    any_GT = False
    for line in fileLines:
        if line_function == 'im_path':
            img_paths.append(line.strip())
            new_im_sample = True
            any_GT = False
            line_function = 'im_num'
            continue
        if line_function == 'im_num':
            pass_indicate = int(line.strip())
            line_function = 'im_GT'
            continue
        if line_function == 'im_GT':
            # invalid face
            if not (int(line.strip().split(' ')[4]) > 0 or int(line.strip().split(' ')[6]) == 1 or int(
                    line.strip().split(' ')[7]) == 1 or int(
                line.strip().split(' ')[8]) > 0 or int(line.strip().split(' ')[9]) == 1):
                any_GT = True
                if new_im_sample:
                    img_GTs.append([(int(line.strip().split(' ')[0]),
                                     int(line.strip().split(' ')[1]),
                                     int(line.strip().split(' ')[2]),
                                     int(line.strip().split(' ')[3]))])
                    new_im_sample = False
                else:
                    img_GTs[len(img_GTs) - 1].append((int(line.strip().split(' ')[0]),
                                                      int(line.strip().split(' ')[1]),
                                                      int(line.strip().split(' ')[2]),
                                                      int(line.strip().split(' ')[3])))
            pass_indicate -= 1
            if pass_indicate == 0:
                if not any_GT:
                    img_paths.pop(-1)
                line_function = 'im_path'
    return img_paths, img_GTs


scales = [48, 24, 12]
threshold_pos = 0.65
threshold_part = 0.4
threshold_neg = 0.3


# calculate two rectangles' iou
# require:
# 1.one rect's parameter:[x,y,w,h]
# 2.another parameter:[x,y,w,h]
# return:
# 1.The iou of rectangles
def from_rects_import_iou(rect1, rect2):
    x = [rect1[0], rect1[0] + rect1[2], rect2[0], rect2[0] + rect2[2]]
    y = [rect1[1], rect1[1] + rect1[3], rect2[1], rect2[1] + rect2[3]]
    if x[1] <= x[2] or x[3] <= x[0] or y[1] <= y[2] or y[3] <= y[0]:
        iou_score = 0
        return iou_score
    x.sort()
    y.sort()
    intersecting_area = (x[2] - x[1]) * (y[2] - y[1])
    if float(rect1[2] * rect1[3] + rect2[2] * rect2[3] - intersecting_area) == 0:
        print(rect1, rect2)
        print('ERROR in iou cal......')
        iou_score = 0
    else:
        iou_score = float(intersecting_area) / float(rect1[2] * rect1[3] + rect2[2] * rect2[3] - intersecting_area)
    return iou_score


# create relevant rectangles base on ground truth
# require:
# 1.ground truth rectangle:[x,y,w,h]
# 2.The interval of neighbouring rectangle
# 3.the max number of output rectangle
# return:
# 1.The list of positive rectangle
# 2.The list of partial rectangle
# 3.The list of negative rectangle
def from_GT_import_relative_rect(rect_GT, img_size, strides=1, num_limit=None):
    rect_pos = []
    rect_neg = []
    rect_part = []
    for scale in scales:
        center_rect = [rect_GT[0] + rect_GT[2] / 2 - scale / 2,
                       rect_GT[1] + rect_GT[3] / 2 - scale / 2,
                       scale, scale]
        max_iou_score = from_rects_import_iou(rect_GT, center_rect)
        if max_iou_score < threshold_pos:
            continue
        moving_rect = copy.deepcopy(center_rect)
        offset = [0, 0]
        strides_x = 1 * strides
        strides_y = 1 * strides
        quadrant = 1
        asymmetric_compensate_x = 0
        asymmetric_compensate_y = 0
        while True:
            overstep_boundary = True if moving_rect[0] < 0 or moving_rect[1] < 0 or moving_rect[0] + moving_rect[2] > \
                                        img_size[0] or moving_rect[1] + moving_rect[3] > img_size[1] else False
            iou_score = from_rects_import_iou(rect_GT, moving_rect)
            offset[0] += strides_x
            if iou_score >= threshold_pos:
                if not overstep_boundary:
                    rect_pos.append(copy.deepcopy(moving_rect))
            elif iou_score >= threshold_part:
                if not overstep_boundary:
                    rect_part.append(copy.deepcopy(moving_rect))
            elif iou_score < threshold_neg:
                if not overstep_boundary:
                    rect_neg.append(copy.deepcopy(moving_rect))
                if moving_rect[0] == center_rect[0] + asymmetric_compensate_x:
                    offset[1] = 0
                    if quadrant == 1:
                        strides_x = -1 * strides
                        strides_y = 1 * strides
                        quadrant += 1
                        asymmetric_compensate_x = -1
                        asymmetric_compensate_y = -1
                    elif quadrant == 2:
                        strides_x = -1 * strides
                        strides_y = -1 * strides
                        quadrant += 1
                        asymmetric_compensate_x = -1
                        asymmetric_compensate_y = 0
                    elif quadrant == 3:
                        strides_x = 1 * strides
                        strides_y = -1 * strides
                        quadrant += 1
                        asymmetric_compensate_x = 0
                        asymmetric_compensate_y = 0
                    else:
                        break
                offset[0] = 0
                offset[1] += strides_y
            moving_rect[0] = center_rect[0] + offset[0] + asymmetric_compensate_x
            moving_rect[1] = center_rect[1] + offset[1] + asymmetric_compensate_y
    rect_list = [rect_pos, rect_part, rect_neg]
    num_rect = [len(rect_pos), len(rect_part), len(rect_neg)]
    if num_limit is None:
        max_num = min(num_rect)
    else:
        max_num = num_limit
    for rects in rect_list:
        if len(rects) == max_num:
            continue
        num_pop = len(rects) - max_num
        if num_pop <= 0:
            continue
        for i in range(num_pop):
            index = np.random.randint(len(rects))
            rects.pop(index)
    return rect_list


# supply negative sample
# require:
# 1.image width
# 2.image height
# 3.The rectangle's list of negative
# 4.img_GTs: like [(rect1 of img1),(rect2 of img1)...]
# 5.ratio of supply sample
# return:
# 1.complete negative sample
def from_relevant_rect_import_complete_samples(im_width, im_height, rect_negs, img_GTs, ratio):
    num_neg_addition = len(rect_negs) * (ratio - 1)
    scale = rect_negs[0][2]

    curr_num = len(rect_negs)
    rect_negs = rect_negs[0:int(curr_num * (scale - 12) / 240)]
    num_neg_addition += curr_num - len(rect_negs)

    for rect_neg in rect_negs:
        for im_GT in img_GTs:
            if from_rects_import_iou(im_GT, rect_neg) >= threshold_neg:
                rect_negs.pop(rect_negs.index(rect_neg))
                num_neg_addition += 1
                break
    knock_num = 5 * num_neg_addition
    while num_neg_addition > 0:
        knock_num -= 1
        if knock_num == 0 or im_width <= scale or im_height <= scale:
            print('lock death...')
            break
        x = np.random.randint(im_width - scale)
        y = np.random.randint(im_height - scale)
        overlap_sample = False
        for im_GT in img_GTs:
            if from_rects_import_iou(im_GT, [x, y, scale, scale]) >= threshold_neg:
                overlap_sample = True
                break
        if not overlap_sample:
            num_neg_addition -= 1
            rect_negs.append([x, y, scale, scale])
    return rect_negs


# create complete sample include positives parts and negatives from single image
# require:
# 1.The single image path
# 2.ground truth like:[(x,y,w,h)...]
# 3.The ratio of neg/pos
# 4.The zoom rate of image pyramid
# return
# 1.complete sample.format:sample--different scale--image array
#                                                 |_different GT--positive sample set--different position rect:[x,y,w,h]
#                                                               |_partial sample set--different position rect:[x,y,w,h]
#                                                               |_negative sample set--different position rect:[x,y,w,h]
#                                                               |_GT position:[x,y,w,h]
def from_im_GTs_import_random_data_set_online(img_path, img_GTs, ratio, scale_rate, set_type):
    im = cv2.imread(join(join(PATH, 'WIDER_%s/images' % set_type), img_path))
    im_rect = copy.deepcopy(im)
    im_scale = copy.deepcopy(im)
    im_width = im.shape[1]
    im_height = im.shape[0]
    max_side = 0
    for img_GT in img_GTs:
        if img_GT[2] > max_side:
            max_side = img_GT[2]
        if img_GT[3] > max_side:
            max_side = img_GT[3]
    rect_sample_scale = []
    while max_side > scales[-1]:
        rect_sample = []
        match_sample = False
        for img_GT in img_GTs:
            rect_poses, rect_parts, rect_negs = from_GT_import_relative_rect(img_GT, (im_width, im_height), strides=2,
                                                                             num_limit=6)
            if len(rect_negs) == 0:
                continue
            match_sample = True
            rect_negs = from_relevant_rect_import_complete_samples(im_width, im_height, rect_negs, img_GTs, ratio)
            rect_sample.append([rect_poses, rect_parts, rect_negs, img_GT])
            # show test
            # for rect in rect_poses:
            #     im_rect = cv2.rectangle(im_rect, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]),
            #                             (0, 0, 255))
            # cv2.imshow('test', im_rect)
            # cv2.waitKey()
        if match_sample:
            rect_sample_scale.append([im_scale, rect_sample])
        # zoom
        max_side /= scale_rate
        im_width = int(np.round(im_width / scale_rate))
        im_height = int(np.round(im_height / scale_rate))
        im_scale = cv2.resize(im_scale, (im_width, im_height))
        im_rect = copy.deepcopy(im_scale)
        zoom_GT = []
        for img_GT in img_GTs:
            zoom_GT.append((int(np.round(img_GT[0] / scale_rate)),
                            int(np.round(img_GT[1] / scale_rate)),
                            int(np.round(img_GT[2] / scale_rate)),
                            int(np.round(img_GT[3] / scale_rate))))
        img_GTs = copy.deepcopy(zoom_GT)
    return rect_sample_scale


# save sample as tfrecords
# require:
# 1.images' path:[path1,path2...]
# 2.ground truth:[[(rect1 of img1),(rect2 of img1)...],[(rect1 of img2),(rect2 of img2)...]...]
# 3.The type of sample:train or val
# return:
def save_samples(img_paths, img_GTs, sample_type):
    writer_12_pos = tf.python_io.TFRecordWriter('./data/%s_12_pos.tfrecords' % sample_type)
    writer_24_pos = tf.python_io.TFRecordWriter('./data/%s_24_pos.tfrecords' % sample_type)
    writer_48_pos = tf.python_io.TFRecordWriter('./data/%s_48_pos.tfrecords' % sample_type)
    writer_12_neg = tf.python_io.TFRecordWriter('./data/%s_12_neg.tfrecords' % sample_type)
    writer_24_neg = tf.python_io.TFRecordWriter('./data/%s_24_neg.tfrecords' % sample_type)
    writer_48_neg = tf.python_io.TFRecordWriter('./data/%s_48_neg.tfrecords' % sample_type)
    writer_12_part = tf.python_io.TFRecordWriter('./data/%s_12_part.tfrecords' % sample_type)
    writer_24_part = tf.python_io.TFRecordWriter('./data/%s_24_part.tfrecords' % sample_type)
    writer_48_part = tf.python_io.TFRecordWriter('./data/%s_48_part.tfrecords' % sample_type)
    writer_12 = [writer_12_pos, writer_12_part, writer_12_neg]
    writer_24 = [writer_24_pos, writer_24_part, writer_24_neg]
    writer_48 = [writer_48_pos, writer_48_part, writer_48_neg]
    num = 0
    for i in range(len(img_paths)):
        rect_sample_scale = from_im_GTs_import_random_data_set_online(img_paths[i], img_GTs[i], 3, 1.2, sample_type)
        num = wirte_TF_records.from_rect_sample_import_tfrecords(rect_sample_scale, writer_48, writer_24, writer_12,
                                                                 num)
        if i % 100 == 0:
            print('save the %d images' % i)
    print('the 48 scale pos rect:%d' % num)
    for i in range(3):
        writer_12[i].close()
        writer_24[i].close()
        writer_48[i].close()


# def from_celebA_import_landmark():
#     landmarks_PATH = '/home/jr/Downloads/celebA/list_landmarks_celeba.txt'
#     bbox_PATH = '/home/jr/Downloads/celebA/list_bbox_celeba.txt'
#     img_PATH='/media/jr/Re/celebA_data/'
#     with open(bbox_PATH, 'r') as bbox_list:
#         b_lines = bbox_list.readlines()
#     with open(landmarks_PATH, 'r') as landmarks_list:
#         l_lines = landmarks_list.readlines()
#     a = 1


if __name__ == '__main__':
    # sample_type = 'val'
    # data_list = from_file_import_datalist(sample_type)
    # img_paths, img_GTs = from_datalist_import_im_GT(data_list)
    # save_samples(img_paths, img_GTs, sample_type)
    from_celebA_import_landmark()

    a = 1
