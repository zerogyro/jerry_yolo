from nets.yolov3 import yolo_body
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import config.config_param as cfg
import numpy as np
from nets.yolo_head import yolo_head
from timeit import default_timer as timer


model_path = '/home/jerry/PycharmProjects/jerry_yolo/log/yolov3_32.3341.h5'
img_path = "/home/jerry/PycharmProjects/yolov3demo/VOC2012/JPEGImages/2007_009348.jpg"




if __name__ == '__main__':
    input = tf.keras.layers.Input(shape=(416, 416, 3))
    model = yolo_body(input, 3, 20)
    model.load_weights(model_path)
    img = Image.open(img_path)
    image_width, image_height = img.size
    input_width, input_height = cfg.input_shape
    scale = min(input_width / image_width, input_height / image_height)
    new_width = int(image_width * scale)
    new_height = int(image_height * scale)

    image = img.resize((new_width, new_height), Image.BICUBIC)
    new_image = Image.new('RGB', cfg.input_shape, (128, 128, 128))
    new_image.paste(image, ((input_width - new_width) // 2, (input_height - new_height) // 2))


    image_data = np.array(new_image, dtype=np.float32)
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # 增加batch的维度
    # plt.imshow(img)
    # plt.show()
    print(image_data.shape)

    start = timer()
    output = model(image_data)
    print(output[0].shape)
    boxes = output[0][...,0:4]
    print(boxes)
    end = timer()
    print("use_time:{:.2f}s".format(end - start))

    # print(a[0].shape)
    # print(a[1].shape)
    # print(a[2].shape)