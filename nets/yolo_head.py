import tensorflow as tf
import numpy as np
import config.config_param as cfg

anchors = cfg.anchors
anchor_masks = cfg.anchor_masks


def yolo_head(y_pred, anchors, calc_loss=False):
    """
    yolo_head split raw prediction from yolo_body into x,y,w,h and bounding box then process with normalization
    and create grid of three sizes
    :param y_pred: raw prediction from yolo_body
    :param anchors: three different anchor box from cfg
    :param calc_loss: True when train, False when predict
    :return: bounding box from prediction and grid
    """
    grid_size = tf.shape(y_pred)[1]

    # reshape_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))

    box_xy, box_wh, confidence, class_probs = tf.split(y_pred, (2, 2, 1, cfg.num_classes), axis=-1)
    # i.e. box_xy.shape = batch_size, grid, grid, num_bbox, xy
    # (8, 13, 13, 3, 2)
    # (8, 26, 26, 3, 2)
    # (8, 52, 52, 3, 2)

    # sigmoid let result lie between [0,1]
    box_xy = tf.sigmoid(box_xy)
    confidence = tf.sigmoid(confidence)
    class_probs = tf.sigmoid(class_probs)

    # create grid

    grid_y = tf.tile(tf.reshape(tf.range(grid_size), [-1, 1, 1, 1]), [1, grid_size, 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(grid_size), [1, -1, 1, 1]), [grid_size, 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], axis=-1)
    grid = tf.cast(grid, tf.float32)

    # -----------------------------normalization x,y,w,h-------------------------------------------------------------
    # cell is offset from the top left corner of the image by (c x , c y ) and the bounding box prior has width and
    # height p w , p h
    # bx = σ(tx ) + cx
    # by = σ(ty ) + cy
    # bw = pwe^tw
    # bh = phe^th
    # ----------------------------------------------------------------------------------------------------------------
    box_xy = (box_xy + grid) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors / cfg.input_shape

    pred_xywh = tf.concat([box_xy, box_wh], axis=-1)  # original xywh for loss

    if calc_loss:
        return pred_xywh, grid

    return box_xy, box_wh, confidence, class_probs
