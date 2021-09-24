from PIL import Image, ImageFont, ImageDraw
import config.config_param as cfg
import numpy as np
import colorsys

img_path = "/home/jerry/PycharmProjects/yolov3demo/VOC2012/JPEGImages/2011_005517.jpg"
real_bbox = [[181, 186, 271, 404],
             [140, 240, 206, 385],
             [48, 182, 138, 395],
             [124, 130, 212, 290]]

pred_bbox = [[99.02229, 89.78006, 331.77008, 242.32076],
             [91.62183, 65.99755, 414.151,  221.7074],
             [92.96844, 107.00315, 409.45706, 231.34407],
             [119.188484, 83.117386, 439.61285, 258.13647],
             [239.10184, 137.38847, 397.86993, 195.5335],
             [251.79779, 150.64944, 382.277,  204.6709]]
real_classes = [14, 14, 14, 14]
pred_classes = [14,14,14,14,14,14]


def draw_orig(image):
    font = ImageFont.load_default()
    # print(image.size)
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in enumerate(pred_classes):
        classes = cfg.class_names[c]
        box = pred_bbox[i]
        top, left, bottom, right = box
        draw = ImageDraw.Draw(image)
        draw.rectangle([top, left, bottom, right], width=thickness)
        # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)])

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
    # image = Image.open(img_path)
    # draw_pred(image)
    # image.show()
    hsv_tuples = [(x / 20, 1., 1.)
                  for x in range(20)]
    print(hsv_tuples)
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    print(colors)
