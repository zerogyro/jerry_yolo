import tensorflow as tf
import config.config_param as cfg
from data_annotation import datareader
from nets.yolo_loss import YoloLoss
from nets.yolov3 import yolo_body

import os
from tensorflow.keras import optimizers, callbacks, metrics
from tensorflow.keras.optimizers.schedules import PolynomialDecay


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # gpus = tf.config.experimental.list_physical_devices("GPU")
    # if gpus:
    #     for gpu in gpus:
    #         tf.config.experimental.set_memory_growth(gpu, True)

    # 读取数据
    #reader = datareader(cfg.data_path, cfg.input_shape, cfg.batch_size)
    train_data = datareader.generate()
    #print(len(list(train_data)))
    validation_data = datareader.generate()
    # train_steps = len(list(train_data)) // cfg.batch_size
    # validation_steps = len(list(validation_data)) // cfg.batch_size

    # print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(list(train_data)),
    #                                                                            len(list(validation_data)),
    #                                                                            cfg.batch_size))

    optimizer = optimizers.Adam(learning_rate=cfg.learn_rating)
    yolo_loss = [YoloLoss(cfg.anchors[mask]) for mask in cfg.anchor_masks]

    train_by_fit(optimizer, yolo_loss, train_data, 20, validation_data, 20)


def train_by_fit(optimizer, loss, train_data, train_steps, validation_data, validation_steps):
    cbk = [
        callbacks.ReduceLROnPlateau(verbose=1),
        callbacks.EarlyStopping(patience=10, verbose=1),
        callbacks.ModelCheckpoint('./log/yolov3_{val_loss:.04f}.h5', save_best_only=True, save_weights_only=True)
    ]

    input = tf.keras.layers.Input(shape=(416, 416, 3))
    model = yolo_body(input, 3, 20)
    #model = yolo_body()
    model.compile(optimizer=optimizer, loss=loss)

    # initial_epoch用于恢复之前的训练
    # model.fit(train_data,
    #           steps_per_epoch=max(1, train_steps),
    #           validation_data=validation_data,
    #           validation_steps=max(1, validation_steps),
    #           epochs=cfg.epochs,
    #           callbacks=cbk)
    model.fit_generator(
        train_data,
        steps_per_epoch=train_steps,
        validation_data=validation_data,
        validation_steps=validation_steps,
        epochs=cfg.epochs,
        #initial_epoch=Freeze_epoch,
        callbacks=cbk)
    #model.save_weights('log' + 'last1.h5')

if __name__ == '__main__':
    main()
