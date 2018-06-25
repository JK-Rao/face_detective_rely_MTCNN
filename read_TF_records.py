from __future__ import absolute_import, division, print_function
import tensorflow as tf
import cv2

NUM_CLASSES = 2


# decode data
def read_and_decode(filename_queue, scale):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'im_raw': tf.FixedLenFeature([], tf.string),
        'rect_a_raw': tf.FixedLenFeature([], tf.string),
        'rect_GT_raw': tf.FixedLenFeature([], tf.string)
    })
    label = tf.cast(features['label'], tf.int32)
    image = tf.decode_raw(features['im_raw'], tf.uint8)
    rect_a = tf.decode_raw(features['rect_a_raw'], tf.int32)
    rect_GT = tf.decode_raw(features['rect_GT_raw'], tf.int32)
    #
    image.set_shape([scale * scale * 3])
    image = tf.reshape(image, [scale, scale, 3])
    # image = tf.cast(image, tf.int64)
    rect_a.set_shape([4])
    rect_a = tf.reshape(rect_a, [4])
    rect_GT.set_shape([4])
    rect_GT = tf.reshape(rect_GT, [4])

    return label, image, rect_a, rect_GT


# get sample from tfrecords
# require:
# 1.data source:train or validation
# 2.The size of batch
# 3.The number of epochs
# 4.The scale of data:48 24 or 12
# 5.sample type:pos part or neg
# return:
# 1.sample's label(one hot)
# 2.The image of sample
# 3.anchor rectangle:[x,y,w,h]
# 4.ground truth rectangle:[x,y,w,h]
def from_tfrecords_import_samples(data_set, batch_size, num_epochs, scale, sample_type):
    if not num_epochs:
        num_epochs = None
    if data_set == 'train':
        file = './data/train_%d_%s.tfrecords' % (scale, sample_type)
    else:
        file = './data/val_%d_%s.tfrecords' % (scale, sample_type)

    # with tf.name_scope('input') as scope:
    filename_queue = tf.train.string_input_producer([file], num_epochs=num_epochs)
    label, image, rect_a, rect_GT = read_and_decode(filename_queue, scale)
    labels, images, rect_as, rect_GTs = tf.train.shuffle_batch([label, image, rect_a, rect_GT],
                                                               batch_size=batch_size,
                                                               num_threads=64,
                                                               capacity=1000 + 3 * batch_size,
                                                               min_after_dequeue=1000
                                                               )
    lab = tf.reshape(labels, [batch_size, 1])
    ind = tf.expand_dims(tf.range(0, batch_size), 1)
    concated = tf.concat([ind, lab], 1)
    ont_hot_labs = tf.sparse_to_dense(concated, [batch_size, NUM_CLASSES], 1.0)
    return ont_hot_labs, images, rect_as, rect_GTs


# self test
if __name__ == '__main__':
    ont_hot_labs, images, rect_as, rect_GTs = from_tfrecords_import_samples('val', 64, None, 12, 'pos')
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        lab_data, img_data, rect_a_data, rect_GT_data = sess.run(
            [ont_hot_labs, images, rect_as, rect_GTs])

        for im_index in range(img_data.shape[0]):
            cv2.imshow('test', img_data[im_index])
            if cv2.waitKey(500) == ord('q'):
                break
        coord.request_stop()
        coord.join(threads)
    sess.close()
    a = 2
