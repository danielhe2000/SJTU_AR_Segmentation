import tensorflow as tf
import numpy as np


def xentropy(prediction, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
    return cost


def mean_iou(input, target, classes):
    """Returns the mean IoU of the prediction
    Args:
        input: the prediction produced by the network, 2d array
        target: the ground truth, 2d array
        classes: total number of classes
    Returns:
        The mean IoU of the ground truth and the prediction
    """
    miou = 0
    for i in range(classes):
        intersection = np.logical_and(target == i, input == i)
        # print(intersection.any())
        union = np.logical_or(target == i, input == i)
        temp = np.sum(intersection) / np.sum(union)
        miou += temp
    return  miou/classes
