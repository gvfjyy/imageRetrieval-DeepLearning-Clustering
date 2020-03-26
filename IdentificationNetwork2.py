import cv2
import tkinter as tk
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
import pickle

def build_graph2(top_k, charset_size):

    #输入
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    images = tf.placeholder(dtype=tf.float32, shape=[None,1024], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='train_flag')

    #结构
    with tf.device('/cpu:0'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training}):
            fc1 = slim.fully_connected(slim.dropout(images, 1.0), 2048,
                                       activation_fn=tf.nn.relu, scope='fc1')
            fc2 = slim.fully_connected(slim.dropout(fc1, keep_prob), 1024,
                                       activation_fn=tf.nn.relu, scope='fc2')
            fc3 = slim.fully_connected(slim.dropout(fc2, keep_prob), 2048,
                                       activation_fn=tf.nn.relu, scope='fc3')

            logits = slim.fully_connected(slim.dropout(fc3, keep_prob), charset_size, activation_fn=None,
                                          scope='logits')
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))
        output=tf.arg_max(logits,1)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            loss = control_flow_ops.with_dependencies([updates], loss)

        global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)
        probabilities = tf.nn.softmax(logits)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        merged_summary_op = tf.summary.merge_all()

        prediceted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
        accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

    return {
        'fc1':fc1,
        'images': images,
        'labels': labels,
        'keep_prob': keep_prob,
        'top_k': top_k,
        'global_step': global_step,
        'train_op': train_op,
        'loss': loss,
        'is_training': is_training,
        'accuracy': accuracy,
        'accuracy_top_k': accuracy_in_top_k,
        'merged_summary_op': merged_summary_op,
        'predicted_distribution': probabilities,
        'predicted_index_top_k': predicted_index_top_k,
        'predicted_val_top_k': prediceted_val_top_k
    }