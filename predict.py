from nets.yolov3 import yolo_body
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import config.config_param as cfg
import numpy as np
from nets.yolo_utils import parse_yolov3_output
from nets.yolo_head import yolo_head
from timeit import default_timer as timer
from PIL import Image, ImageFont, ImageDraw
import colorsys

model_path = '/home/jerry/PycharmProjects/jerry_yolo/log/yolov3_32.3341.h5'
img_path = "/home/jerry/PycharmProjects/yolov3demo/VOC2012/JPEGImages/2011_005517.jpg"


# @staticmethod
def process_image(image_path):
    #assert isinstance(image_path, object)
    img = Image.open(image_path)
    image_width, image_height = img.size
    input_width, input_height = cfg.input_shape

    # resize image and add gray margin to meet input shape

    scale = min(input_width / image_width, input_height / image_height)
    new_width = int(image_width * scale)
    new_height = int(image_height * scale)
    image = img.resize((new_width, new_height), Image.BICUBIC)
    new_image = Image.new('RGB', cfg.input_shape, (128, 128, 128))
    new_image.paste(image, ((input_width - new_width) // 2, (input_height - new_height) // 2))

    # normalize and expend dimension for batch
    image_data = np.array(new_image, dtype=np.float32)
    image_data /= 255
    image_data = np.expand_dims(image_data, 0)
    return (image_height,image_width), image_data




def draw_pred(image,pred_bbox,pred_classes):
    font = ImageFont.load_default()
    # print(image.size)
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in enumerate(pred_classes):
        classes = cfg.class_names[c]
        box = pred_bbox[i]
        top, left, bottom, right = box
        draw = ImageDraw.Draw(image)
        draw.rectangle([left, top, right, bottom], width=thickness)
        # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)])


if __name__ == '__main__':
    model_input = tf.keras.layers.Input(shape=(416, 416, 3))
    model = yolo_body(model_input, 3, 20)
    model.load_weights(model_path)

    image = Image.open(img_path)
    img_shape, image_data = process_image(img_path)
    output = model(image_data)
    # #print(output[0].shape)
    boxes, scores, classes = parse_yolov3_output(output,img_shape)
    print(boxes)
    print(scores)
    print(classes)
    #image = detect_image(image)
    #image.show()
    draw_pred(image,boxes,classes)
    image.show()

    # print('Found {} boxes for {}'.format(len(boxes), 'img'))





