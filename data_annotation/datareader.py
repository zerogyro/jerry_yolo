from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import config.config_param as cfg

check = '/home/jerry/PycharmProjects/yolov3demo/VOC2012/JPEGImages/2011_005920.jpg 10,40,267,323,14 200,21,312,259,14 303,18,422,323,14 23,10,99,62,14'
check2 = '/home/jerry/PycharmProjects/yolov3demo/VOC2012/JPEGImages/2008_008051.jpg'
max_boxes = 100


def read_and_split():
    """
    split train and valid data from data_path
    """
    with open(cfg.data_path, "r") as f:
        files = f.readlines()
    split = int(cfg.valid_rate * len(files))
    train = files[split:]
    valid = files[:split]
    return train, valid


def get_data(annotation_line):
    """
    resize image data to input shape and correct bounding boxes to real boxes
    """
    input_shape = cfg.input_shape
    line = annotation_line.split()
    image = Image.open(line[0])

    box = np.array([list(map(int, xys.split(','))) for xys in line[1:]])
    image_width, image_height = image.size
    input_width, input_height = input_shape

    scale = min(input_width / image_width, input_height / image_height)

    new_width = int(image_width * scale)
    new_height = int(image_height * scale)
    new_shape = (new_width, new_height)
    image = image.resize(new_shape)

    # add gray bar
    new_image = Image.new('RGB', input_shape, (128, 128, 128))
    new_image.paste(image, ((input_width - new_width) // 2, (input_height - new_height) // 2))
    image = np.asarray(new_image) / 255

    dx = (input_width - new_width) / 2
    dy = (input_height - new_height) / 2

    # correct bbox coord (xmin,ymin xmax,ymax ---> corrected coordinate)

    box_data = np.zeros([max_boxes, 5], dtype='float32')
    if len(box) > max_boxes:
        box = box[:max_boxes]
    box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
    box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
    box_data[:len(box)] = box
    return image, box_data


def process_true_boxes(box_data):
    """
    transfer bounding box to feature layer: #xmin #ymin #xmax #ymax -->
    [(8, 13, 13, 3, 25),
    (8, 26, 26, 3, 25),
    (8, 52, 52, 3, 25)]
    """

    input_shape = cfg.input_shape
    true_boxes = np.array(box_data, dtype='float32')
    x_shape = np.array(input_shape, dtype='int32')

    # get center point and width and height
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    # true box x y w h c after normalization
    # true box shape (8,100,5)
    true_boxes[..., 0:2] = boxes_xy / input_shape
    true_boxes[..., 2:4] = boxes_wh / input_shape


    # Attention!!!!!!!
    # need to change when change structure of yolo body

    # 13,13 26,26 52,52
    grid_shape = [x_shape // [32, 16, 8][i] for i in range(3)]

    # y_true shape [(8, 13, 13, 3, 25), (8, 26, 26, 3, 25), (8, 52, 52, 3, 25)]
    y_true = [np.zeros((cfg.batch_size, grid_shape[i][0], grid_shape[i][1], 3, 5 + 20), dtype='float32') for i in
              range(3)]
    #check shape
    # a = [(y_true[i].shape) for i in range(3)]
    # print(a)

    # anchors shape (1,9,2) cfg.anchors shape(9,2)
    anchors = np.expand_dims(cfg.anchors, 0)
    anchors_rightdown = cfg.anchors / 2.
    anchors_leftup = -anchors_rightdown
    # valid mask stores boolean from box_data i.e whether contains box
    valid_mask = box_data[..., 0] > 0


    for b in range(cfg.batch_size):
        # get w h of every box in batch_size
        wh = boxes_wh[b,valid_mask[b]]
        # (_,1,2)
        wh = np.expand_dims(wh,-2)

        box_rightdown = wh/2.
        box_leftup = -box_rightdown

        intersect_leftup = np.maximum(box_leftup,anchors_leftup)
        intersect_rightdown = np.minimum(box_rightdown,anchors_rightdown)
        intersect_wh = np.maximum(intersect_rightdown-intersect_leftup,0.)
        intersect_area = intersect_wh[...,0] * intersect_wh[...,1]

        box_area = wh[...,0]*wh[...,1]
        anchor_area = cfg.anchors[...,0] * cfg.anchors[...,1]
        #calculate max iou
        iou = intersect_area/(box_area+anchor_area-intersect_area)

        #best_anchor: bounding box most closed anchor box of each image
        best_anchor = np.argmax(iou,axis = -1)


        for key, value in enumerate(best_anchor):
            # for each bounding box, check 3 layer of grid and add to y_true
            for n in range(cfg.num_bbox):
                if value in cfg.anchor_masks[n]:
                    # i = x*13, j = y*13 sub pixel x
                    i = np.floor(true_boxes[b, key, 0] * grid_shape[n][1]).astype('int32')
                    j = np.floor(true_boxes[b, key, 1] * grid_shape[n][0]).astype('int32')

                    k = cfg.anchor_masks[n].index(value)
                    c = true_boxes[b,key,4].astype('int32')

                    y_true[n][b, j, i, k, 0:4] = true_boxes[b, key, 0:4]
                    y_true[n][b, j, i, k, 4] = 1  # confidence 1
                    y_true[n][b, j, i, k, 5 + c] = 1  # one-hot encode
    # print(y_true[0].shape)
    # print(y_true[1].shape)
    # print(y_true[2].shape)
    return y_true



def generate():
    n = len(train)

    i = 0
    while i<16:
        image_data = []
        box_data = []
        for b in range(cfg.batch_size):

            image, bbox = get_data(train[i])
            image_data.append(image)
            box_data.append(bbox)
            i +=1
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        box_data = process_true_boxes(box_data)

        yield image_data, box_data


if __name__ == '__main__':
    train,valid = read_and_split()

    # image, box = get_data(check)
    # # plt.imshow(image)
    # # plt.show()
    #
    train_data = generate()
    train_data = list(train_data)
    print(len(train_data))
    print((train_data[0])[0].shape)
    print(train_data[0][1][0].shape)
    print(train_data[0][1][1].shape)
    print(train_data[0][1][2].shape)

    # y_true = process_true_boxes(a)
    # print(y_true)

