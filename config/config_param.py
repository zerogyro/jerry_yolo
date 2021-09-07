import numpy as np

# input data
valid_rate = 0.1
data_path = '/home/jerry/PycharmProjects/yolodemo/2012_train.txt'
input_shape = (416, 416)

num_classes = 20

anchors = np.array([(10, 13), (16, 30), (33, 23),
                    (30, 61), (62, 45), (59, 119),
                    (116, 90), (156, 198), (373, 326)],
                   np.float32)

# 先验框对应索引
anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
num_bbox = 3
#   training constant
valid_rate = 0.1
batch_size = 8

anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]


ignore_thresh = 0.5

# 训练信息
epochs = 50
# 学习率
learn_rating = 1e-5
# iou忽略阈值
ignore_thresh = 0.5
iou_threshold = 0.3