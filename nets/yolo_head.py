from nets.yolov3 import test_pred
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
num_classes =20
# 先验框信息
anchors = np.array([(10, 13), (16, 30), (33, 23),
                    (30, 61), (62, 45), (59, 119),
                    (116, 90), (156, 198), (373, 326)],
                   np.float32)

# 先验框对应索引
anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    feats = K.constant(feats)
    #print(feats.shape)
    #print(feats.shape)
    num_anchors = len(anchors)
    # ---------------------------------------------------#
    #   [1, 1, 1, num_anchors, 2]
    # ---------------------------------------------------#
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
    #input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    #print(anchors_tensor)
    # ---------------------------------------------------#
    #   获得x，y的网格
    #   (13, 13, 1, 2)
    # ---------------------------------------------------#
    grid_shape = K.shape(feats)[1:3]  # height, width
    # print('grid_ shape')
    # print(grid_shape)
    #grid_shape = K.shape(feats)
    #print(grid_shape)
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    # print(grid_x.shape)
    # print(grid_y.shape)
    grid = K.concatenate([grid_x, grid_y])
    # print(type(grid))
    # print(grid)
    grid = K.cast(grid, K.dtype(feats))
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
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    #print(feats.shape)
    #print(feats[...,2:4].shape)
    #print(input_shape)
    a = K.cast(input_shape[..., ::-1], K.dtype(feats))
    #print(a)
    # ---------------------------------------------------#
    #   将预测值调成真实值
    #   box_xy对应框的中心点
    #   box_wh对应框的宽和高
    # ---------------------------------------------------#
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[..., ::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[..., ::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # ---------------------------------------------------------------------#
    #   在计算loss的时候返回grid, feats, box_xy, box_wh
    #   在预测的时候返回box_xy, box_wh, box_confidence, box_class_probs
    # ---------------------------------------------------------------------#
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

def shape_check(feats):
    print(feats.shape)
if __name__ == '__main__':
    a = test_pred


    # (3,1,13,13,75)
    # (3,1,26,26,75)
    # (3,1,52,52,75)
    #print(K.shape(a[0])[1:3]*32)
    #print(K.dtype(a[0]))
    input_shape = (416,416)
    input_shape = K.constant(input_shape)
    grid, raw_pred, pred_xy, pred_wh = yolo_head(a[1],
                                                  anchors[anchor_masks[0]], num_classes, input_shape, calc_loss=True)
    #a = shape_check(a[1])
    print('----------------------------------------------------------------------')
    print(grid.shape)
    print(raw_pred.shape)
    print(pred_xy.shape)
    print(pred_wh.shape)

    # print(raw_pred[0][0][0][0])
    # print(pred_wh[0][0][0][0])