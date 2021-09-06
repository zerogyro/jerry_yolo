import tensorflow as tf
import numpy as np
import config.config_param as cfg

anchors = cfg.anchors
anchor_masks = cfg.anchor_masks


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    feats = tf.constant(feats)
    # print(feats.shape)
    # print(feats.shape)
    num_anchors = len(anchors)
    # ---------------------------------------------------#
    #   [1, 1, 1, num_anchors, 2]
    # ---------------------------------------------------#
    anchors_tensor = tf.reshape(tf.constant(anchors), [1, 1, 1, num_anchors, 2])

    # ---------------------------------------------------#
    #   获得x，y的网格
    #   (13, 13, 1, 2)
    # ---------------------------------------------------#
    grid_shape = tf.shape(feats)[1:3]  # height, width
    # print('grid_ shape')
    # print(grid_shape)
    # grid_shape = tf.shape(feats)
    # print(grid_shape)
    grid_y = tf.tile(tf.reshape(tf.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                     [1, grid_shape[1], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                     [grid_shape[0], 1, 1, 1])
    # print(grid_x.shape)
    # print(grid_y.shape)
    grid = tf.concatenate([grid_x, grid_y])
    # print(type(grid))
    # print(grid)
    grid = tf.cast(grid, tf.dtype(feats))
    # print('---------------------------------grid------------------------------------------')
    # print(grid[10][10])
    # print('---------------------------------gri-----------------------------------')
    # ---------------------------------------------------#
    #   将预测结果调整成(batch_size,13,13,3,85)
    #   85可拆分成4 + 1 + 80
    #   4代表的是中心宽高的调整参数
    #   1代表的是框的置信度
    #   80代表的是种类的置信度
    # ---------------------------------------------------#
    feats = tf.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    # print(feats.shape)
    # print(feats[...,2:4].shape)
    # print(input_shape)
    a = tf.cast(input_shape[..., ::-1], tf.dtype(feats))
    # print(a)
    # ---------------------------------------------------#
    #   将预测值调成真实值
    #   box_xy对应框的中心点
    #   box_wh对应框的宽和高
    # ---------------------------------------------------#
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(grid_shape[..., ::-1], tf.dtype(feats))
    box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(input_shape[..., ::-1], tf.dtype(feats))
    box_confidence = tf.sigmoid(feats[..., 4:5])
    box_class_probs = tf.sigmoid(feats[..., 5:])

    # ---------------------------------------------------------------------#
    #   在计算loss的时候返回grid, feats, box_xy, box_wh
    #   在预测的时候返回box_xy, box_wh, box_confidence, box_class_probs
    # ---------------------------------------------------------------------#
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_head_new(y_pred, anchors, calc_loss=False):
    """
    另外，取名为head是有意义的。因为目标检测大多数分为 - Backbone - Detection head两个部分
    :param y_pred: 预测数据
    :param anchors: 其中一种大小的先验框（总共三种）
    :param calc_loss: 是否计算loss，该函数可以在直接预测的地方用
    :return:
        bbox: 存储了x1, y1 x2, y2的坐标 shape(b, 13, 13 ,3, 4)
        objectness: 该分类的置信度 shape(b, 13, 13 ,3, 1)
        class_probs: 存储了20个分类在sigmoid函数激活后的数值 shape(b, 13, 13 ,3, 20)
        pred_xywh: 把xy(中心点),wh shape(b, 13, 13 ,3, 4)
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


