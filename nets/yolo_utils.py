import tensorflow as tf
import config.config_param as cfg
from nets.yolo_head import yolo_head

from nets.yolov3 import yolo_body
maxboxes=20
score_threshold = 0.05

def parse_yolov3_output(yolo_outputs,image_shape):
    print('image_shape:',image_shape)
    boxes = []
    box_scores = []

    # get_boxes_and_scores(yolo_outputs[0], cfg.anchors[cfg.anchor_masks[0]],image_shape)
    for i in range(3):
        mask_index = cfg.anchor_masks[i]
        _boxes, _boxes_scores = get_boxes_and_scores(yolo_outputs[i], cfg.anchors[mask_index],image_shape)
        boxes.append(_boxes)
        box_scores.append(_boxes_scores)
        # print('_boxes.shape', _boxes.shape)
        #print('_boxes_scores.shape',_boxes_scores.shape)
    boxes = tf.concat(boxes,axis=0)
    box_scores = tf.concat(box_scores,axis=0)


    # print(boxes.shape)
    # print(box_scores.shape)
    # score_threshold
    mask = box_scores >= score_threshold
    max_boxes_tensor = tf.constant(maxboxes,dtype=tf.int32)

    boxes_ = []
    scores_ = []
    classes_ = []
    #need nms
    for c in range(cfg.num_classes):

        # get box_scores >= threshold bounding box and scores
        #print(c)
        class_box = tf.boolean_mask(boxes, mask[:,c])
        class_box_score = tf.boolean_mask(box_scores[:,c],mask[:,c])
        #print(class_box_score)

        # print(class_box.shape)
        # print(class_box_score.shape)
        classes = tf.ones_like(class_box_score,'int32')*c

        boxes_.append(class_box)
        scores_.append(class_box_score)
        classes_.append(classes)
    boxes_ = tf.concat(boxes_, axis=0)
    scores_ = tf.concat(scores_, axis=0)
    classes_ = tf.concat(classes_, axis=0)

    return boxes_ , scores_ , classes_






def get_boxes_and_scores(feats, anchors , image_shape):
    box_xy, box_wh, box_confidence, box_class_prob = yolo_head(feats, anchors, calc_loss=False)
    boxes = correct_boxes(box_xy,box_wh,image_shape)
    # 1,13,13,3,4 -> 507,4
    boxes = tf.reshape(boxes,[-1,4])

    # define box_score
    # 1,13,13,3,20 -> 507,20
    box_scores = box_confidence * box_class_prob
    box_scores = tf.reshape(box_scores, [-1, cfg.num_classes])
    return boxes, box_scores



def correct_boxes(box_xy, box_wh, image_shape):
    # inverse for grid
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    input_shape = tf.cast(cfg.input_shape,tf.float32)
    image_shape = tf.cast(image_shape,tf.float32)

    # change the shape of bounding box back to normal size for original image

    #tf.min(input_shape/image_shape)
    new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx-offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = tf.concat([
        box_mins[...,0:1], # y_min
        box_mins[...,1:2], # x_min
        box_maxes[...,0:1], # y_max
        box_maxes[...,1:2], # x_max
    ], axis=-1)

    # 1,13,13,3,4
    a =  tf.concat([image_shape,image_shape],axis = -1)

    #
    boxes *= tf.concat([image_shape,image_shape],axis = -1)
    return boxes