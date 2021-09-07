import config.config_param as cfg
import numpy as np
from nets.yolov3 import test_pred
from nets.yolo_head import yolo_head_new
import tensorflow as tf
from data_annotation.datareader import *




def box_iou(pred_box, true_box):
    """
    calculate iou between pred_box and true_box
    :return: iou: tensor, shape=(i1, ..., iN, j)
    """
    # 13,13,3,1,4

    pred_box = tf.expand_dims(pred_box, -2)
    pred_box_xy = pred_box[..., 0:2]
    pred_box_wh = pred_box[..., 2:4]
    pred_box_wh_half = pred_box_wh / 2.
    pred_box_leftup = pred_box_xy - pred_box_wh_half
    pred_box_rightdown = pred_box_xy + pred_box_wh_half

    # 1,n,4

    true_box = tf.expand_dims(true_box, 0)
    true_box_xy = true_box[..., 0:2]
    true_box_wh = true_box[..., 2:4]
    true_box_wh_half = true_box_wh / 2.
    true_box_leftup = true_box_xy - true_box_wh_half
    true_box_rightdown = true_box_xy + true_box_wh_half


    intersect_leftup = tf.maximum(pred_box_leftup, true_box_leftup)
    intersect_rightdown = tf.minimum(pred_box_rightdown, true_box_rightdown)

    intersect_wh = tf.maximum(intersect_rightdown - intersect_leftup, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]


    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]

    iou = intersect_area / (pred_box_area + true_box_area - intersect_area)

    return iou


def YoloLoss(anchors):
    def compute_loss(y_true, y_pred):
        """
        generate loss function from y_true and y_pred
        :param y_true: (x,y,w,h,c) that x,y,w,h are ratio to input shape from normalization
        :param y_pred: (x,y,w,h,c) that x,y are offsets from grid cell
        :return: loss sum of x,y, w,h confidence and categories
        """
        input_shape = cfg.input_shape
        grid_shapes = tf.cast(tf.shape(y_pred)[1:3], tf.float32)
        # print(grid_shapes)

        # y_pred: (batch_size, grid, grid, anchors * (x, y, w, h, obj, ...cls))
        pred_xywh, grid = yolo_head_new(y_pred, anchors, calc_loss=True)

        # get pred xy: x_offset, y_offset
        pred_xy = y_pred[..., 0:2]
        pred_wh = y_pred[..., 2:4]
        pred_conf = y_pred[..., 4:5]
        pred_class = y_pred[..., 5:]

        # calculate offset_x and offset_y from y_true
        true_xy = y_true[..., 0:2] * grid_shapes - grid
        # calculate ratio of w, h to anchor box
        true_wh = tf.math.log(y_true[..., 2:4] / anchors * input_shape)
        object_mask = y_true[..., 4:5]
        true_class = y_true[..., 5:]

        # set invalid w,h to zero since log(0) will cause -inf and
        # tf.where is to set condition True to zero and unchange the value if condition is false

        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # box_loss_scale: a parameter that let small box to have higher weights in loss
        # if object detected mostly small, tune it to higher loss

        box_loss_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]

        # find negative sample group
        # 找到负样本群组，第一步是创建一个数组，[]
        ignore_mask = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
        object_mask_bool = tf.cast(object_mask, tf.bool)

        # 对每一张图片计算ignore_mask
        def loop_body(b, ignore_mask):
            # in object_mask_bool, only True value are valid that y_true[...,0:4] are valid

            true_box = tf.boolean_mask(y_true[b, ..., 0:4], object_mask_bool[b, ..., 0])

            # calculate iou from pred_box and true_box
            iou = box_iou(pred_xywh[b], true_box)
            best_iou = tf.reduce_max(iou, axis=-1)
            # if the highest score of iou is still smaller than threshold then consider it as negative sample of the
            # image
            ignore_mask = ignore_mask.write(b, tf.cast(best_iou < cfg.ignore_thresh, tf.float32))
            return b + 1, ignore_mask

        batch_size = tf.shape(y_pred)[0]

        # create tf.while loop
        # args1 loop condition: b<batch_size
        # args2 loop body: loop_body function
        # args3 initial param: b=0, ignore_mask empty
        # lambda b,*args: b<m, *args, b<batch_size

        _, ignore_mask = tf.while_loop(lambda b, ignore_mask: b < batch_size, loop_body, [0, ignore_mask])

        # compress and expend dim to match loss
        ignore_mask = ignore_mask.stack()

        ignore_mask = tf.expand_dims(ignore_mask, -1)

        xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(true_xy,
                                                                                         pred_xy)
        wh_loss = object_mask * box_loss_scale * tf.square(true_wh - pred_wh)
        object_conf = tf.nn.sigmoid_cross_entropy_with_logits(object_mask, pred_conf)
        confidence_loss = object_mask * object_conf + (1 - object_mask) * object_conf * ignore_mask
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(true_class, pred_class)

        # average loss
        xy_loss = tf.reduce_sum(xy_loss) / tf.cast(batch_size, tf.float32)
        wh_loss = tf.reduce_sum(wh_loss) / tf.cast(batch_size, tf.float32)
        confidence_loss = tf.reduce_sum(confidence_loss) / tf.cast(batch_size, tf.float32)
        class_loss = tf.reduce_sum(class_loss) / tf.cast(batch_size, tf.float32)

        return xy_loss + wh_loss + confidence_loss + class_loss

    return compute_loss


