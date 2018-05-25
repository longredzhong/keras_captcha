# coding: utf-8
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import numpy as np
from skimage import io
import os
import os.path
import xlrd
from keras.utils.vis_utils import plot_model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical


def read_mappings():  # 读取标签
    path = os.path.join(os.getcwd(), 'train_change')
    path = os.path.join(path,'mappings.txt')
    file = open(path,encoding='gbk')
    lables = []
    for line in file:
        line=line.strip('\n')
        line = line.split(',')
        tmp = [line[0],line[1]]
        #print tmp
        lables.append(tmp)
    return lables


def read_imaeg(lables, path=os.path.join(os.getcwd(),
                                         'train_change')):  #读取图片并链接标记
    x = []
    y = []
    picnum = len(lables)
    for i in range(picnum):
        img = io.imread(os.path.join(path, lables[i] + '.jpg'), 0)
        x.append(img)
        y.append(lables[i][1])
    return x, y


def format_data(x, y, img_rows=60, img_cols=200,
                num_classes=36):  #格式化数据，输出训练数据,测试数据,检测数据
    labeldict = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6,
        'H': 7,
        'I': 8,
        'J': 9,
        'K': 10,
        'L': 11,
        'M': 12,
        'N': 13,
        'O': 14,
        'P': 15,
        'Q': 16,
        'R': 17,
        'S': 18,
        'T': 19,
        'U': 20,
        'V': 21,
        'W': 22,
        'X': 23,
        'Y': 24,
        'Z': 25,
        '0': 26,
        '1': 27,
        '2': 28,
        '3': 29,
        '4': 30,
        '5': 31,
        '6': 32,
        '7': 33,
        '8': 34,
        '9': 35
    }
    num_classes = 36
    x = np.array(x)
    y_detect = y[8000:]
    y_data = y[:]
    for i in range(len(y)):
        c0 = keras.utils.to_categorical(labeldict[y[i][0]], num_classes)
        c1 = keras.utils.to_categorical(labeldict[y[i][1]], num_classes)
        c2 = keras.utils.to_categorical(labeldict[y[i][2]], num_classes)
        c3 = keras.utils.to_categorical(labeldict[y[i][3]], num_classes)
        c4 = keras.utils.to_categorical(labeldict[y[i][4]], num_classes)
        c = np.concatenate((c0, c1, c2, c3,c4), axis=0)
        y[i] = c
    y = np.array(y)
    x_data = x[:]
    x_train = x[:8000]
    y_train = y[:8000]
    x_test = x[8000:]
    y_test = y[8000:]
    x_detect = x[8000:]

    x_data = x_data.reshape(x_data.shape[0], img_rows, img_cols, 1)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_detect = x_detect.reshape(x_detect.shape[0], img_rows, img_cols, 1)
    x_data = x_data.astype('float32')
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_detect = x_detect.astype('float32')
    x_data /= 255
    x_train /= 255
    x_test /= 255
    x_detect /= 255
    return x_data, y_data, x_train, x_test, x_detect, y_train, y_test, y_detect


def make_model(input_shape, num_classes, model_path):  #创建模型
    model = Sequential()
    # 1 conv
    model.add(
        Conv2D(
            64,
            5,
            5,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01),
            input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    # 2 conv
    model.add(
        Conv2D(
            64,
            5,
            5,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
        ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    # 3 conv
    model.add(
        Conv2D(
            128,
            5,
            5,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
        ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    # 4 conv
    model.add(
        Conv2D(
            128,
            5,
            5,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
        ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    # 1 Dense
    model.add(Flatten())
    model.add(
        Dense(
            200,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
        ))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    # 2 Dense
    model.add(Dense(num_classes * 5, activation='softmax'))
    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer='adam',
        metrics=['accuracy'])
    model.save(model_path)
    return model


def trains(x_train, y_train, x_test, y_test, epochs, batch_size,
           model_path):  #训练
    model = load_model(model_path)
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test))
    model.save(model_path)
    K.clear_session()


def detec(x_detect, y_detect, model_path):  #检测
    model = load_model(model_path)
    pred = model.predict(x_detect, batch_size=50)
    outdict = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1',
        '2', '3', '4', '5', '6', '7', '8', '9'
    ]
    correct_num = 0
    for i in range(pred.shape[0]):
        c0 = outdict[np.argmax(pred[i][:36])]
        c1 = outdict[np.argmax(pred[i][36:36 * 2])]
        c2 = outdict[np.argmax(pred[i][36 * 2:36 * 3])]
        c3 = outdict[np.argmax(pred[i][36 * 3:36 * 4])]
        c4 = outdict[np.argmax(pred[i][36 * 4:])]
        c = c0 + c1 + c2 + c3 +c4
        print(c, y_detect[i])
        if c == y_detect[i]:
            correct_num = correct_num + 1
    print(correct_num)
    print("Test Whole Accurate : ", float(correct_num) / len(pred))  #统计整体正确率
    return float(correct_num) / len(pred)
    K.clear_session()


def main():
    lables = read_mappings()
    x, y = read_imaeg(lables)
    x_data, y_data, x_train, x_test, x_detect, y_train, y_test, y_detect = format_data(
        x, y)
    num_classes = 36
    img_rows, img_cols = 60, 200
    input_shape = (img_rows, img_cols, 1)
    Probability = 0
    seed = 7
    np.random.seed(seed)
    epochs = 50
    batch_size = 10
    model_path = 'my_model.h5'
    model_img_path = 'my_model.png'
    try:
        model = load_model(model_path)
        plot_model(model, to_file=model_img_path,show_shapes=True)
    except:
        model = make_model(input_shape, num_classes, model_path)
        plot_model(model,to_file=model_img_path,show_shapes=True)
    finally:
        while (Probability < 0.9):
            trains(x_train, y_train, x_test, y_test, epochs, batch_size,
                   model_path)
            Probability = detec(x_detect, y_detect, model_path)
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("Test Whole Accurate : ", detec(x_data, y_data, model_path))


if __name__ == '__main__':
    main()
