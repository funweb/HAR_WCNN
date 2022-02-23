#!/usr/bin/env python3

from keras.layers import Dense, LSTM, Bidirectional, Average, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.initializers import he_uniform

from keras import regularizers

from keras.applications import inception_v3

from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, MaxPool1D, GlobalAveragePooling1D, \
    concatenate, Activation
from keras.models import Model



# input_dim: int > 0。
# output_dim: int >= 0。
def get_LSTM(vocabulary_size, output_dim, data_lenght, no_activities):
    model = Sequential(name='LSTM')
    model.add(Embedding(vocabulary_size, output_dim, input_length=data_lenght, mask_zero=True))  # output_dim * input_dim
    model.add(LSTM(output_dim))  # (output_dim + input_dim + 1) * （output_dim * 4）

    model.add(Dense(no_activities, activation='softmax'))  # (input_dim * no_activities + no_activites)
    return model


def WCNN(no_activities, vocabulary_size=188, output_dim=64, data_lenght=2000, kernel_number_base=8, kernel_wide_base=1):
    print('no_activities: %d\n' % (no_activities))
    ip = Input(shape=(data_lenght,))
    emb = Embedding(vocabulary_size,
                    output_dim,
                    # weights=[embedding_matrix],
                    input_length=data_lenght,
                    trainable=True)(ip)

    emb = BatchNormalization()(emb)

    ly_he_uniform_1 = he_uniform(1)
    cnn1 = Conv1D(4*kernel_number_base, 1*kernel_wide_base, padding='same', kernel_initializer=ly_he_uniform_1)(emb)

    cnn1 = Activation('relu')(cnn1)


    ly_he_uniform_2 = he_uniform(2)
    cnn2 = Conv1D(4*kernel_number_base, 3*kernel_wide_base, padding='same', kernel_initializer=ly_he_uniform_2)(emb)
    cnn2 = Activation('relu')(cnn2)


    ly_he_uniform_3 = he_uniform(3)
    cnn3 = Conv1D(2*kernel_number_base, 5*kernel_wide_base, padding='same', kernel_initializer=ly_he_uniform_3)(emb)
    cnn3 = Activation('relu')(cnn3)

    ly_he_uniform_4 = he_uniform(4)
    cnn4 = Conv1D(1 * kernel_number_base, 7*kernel_wide_base, padding='same', kernel_initializer=ly_he_uniform_4)(emb)
    cnn4 = Activation('relu')(cnn4)

    # max_pool_1 = MaxPool1D(pool_size=3, strides=3, padding='same')(emb)  # 不知道这个有用没

    cnn = concatenate([cnn1, cnn2, cnn3, cnn4])

    cnn = GlobalMaxPooling1D()(cnn)
    cnn = BatchNormalization(axis=1)(cnn)

    dcnn1 = Dense(4*kernel_number_base)(cnn)
    dcnn1 = Activation('relu')(dcnn1)
    dcnn1 = Dropout(0.4)(dcnn1)


    dcnn2 = Dense(4*kernel_number_base)(dcnn1)
    dcnn2 = Activation('relu')(dcnn2)


    out = Dense(no_activities, activation='softmax')(dcnn2)
    model = Model(ip, out)

    return model


def embedding_init():
    return keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=7)


def WCNNR(no_activities, vocabulary_size=188, output_dim=64, data_lenght=2000, kernel_number_base=8, kernel_wide_base=1, net_deep_base=1):
    print('no_activities: %d\n' % (no_activities))
    ip = Input(shape=(data_lenght,))
    emb = Embedding(vocabulary_size,
                    output_dim,
                    embeddings_initializer=embedding_init(),
                    # weights=[embedding_matrix],
                    input_length=data_lenght,
                    trainable=True)(ip)

    # emb = BatchNormalization()(emb)

    cnn_total = emb  # 方便下面使用

    for d in range(net_deep_base):
        # 网络的宽度
        wide_kernal = []
        for i in range(kernel_wide_base):
            cnn = Conv1D(filters=kernel_number_base*4, kernel_size=(i + 1) * 2 - 1, padding='same',
                         kernel_initializer=he_uniform(i))(cnn_total)  # 要不要加入激活层, 有待商榷
            wide_kernal.append(cnn)

        if len(wide_kernal) > 1:
            cnn_total = concatenate(wide_kernal)  # 好像默认: , axis=-1
        else:
            cnn_total = wide_kernal[0]

        cnn_total = Activation('relu')(cnn_total)


    cnn_total = GlobalMaxPooling1D()(cnn_total)
    # cnn = BatchNormalization(axis=1)(cnn)

    dcnn1 = Dense(6*kernel_number_base)(cnn_total)
    dcnn1 = Activation('relu')(dcnn1)
    dcnn1 = Dropout(0.4)(dcnn1)

    dcnn2 = Dense(4*kernel_number_base)(dcnn1)
    dcnn2 = Activation('relu')(dcnn2)


    out = Dense(no_activities, activation='softmax')(dcnn2)
    model = Model(ip, out)

    return model



# 自定义优化器
from keras import optimizers

def compileModelcus(model, optimizer_name_flag):
    # model.compile(loss='categorical_crossentropy', optimizer='sgd')
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if optimizer_name_flag == 'sgd':
        # 随机梯度下降（Stochastic gradient descent）
        optimizer_name = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    elif optimizer_name_flag == 'adagrad':
        # 可以自适应的调整学习率
        optimizer_name = optimizers.Adagrad(lr=0.01, epsilon=1e-06)

    elif optimizer_name_flag == 'adadelta':
        # 解决 Adagrad 学习率急速下降问题
        optimizer_name = optimizers.Adadelta(lr=0.001, rho=0.95, epsilon=1e-06)

    elif optimizer_name_flag == 'rms':
        # 解决 Adagrad 学习率急速下降问题
        optimizer_name = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    elif optimizer_name_flag == 'adam':
        # Adam本质上像是带有动量项的RMSprop，不知道用哪个的时候用这个
        optimizer_name = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif optimizer_name_flag == 'adammax':
        optimizer_name = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif optimizer_name_flag == 'nadam':
        # Nadam就是带有Nesterov 动量的Adam RMSprop
        optimizer_name = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    # model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])  # 我需要用 sgd
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer_name, metrics=['accuracy'])  # 论文中用的 adam
    # model.summary()
    return model




import keras.applications.inception_v3


### ---------------------- new ------------------------------
from keras.layers import Conv1D, Multiply, MaxPool1D, RepeatVector, \
    Reshape, Activation, MaxPool1D, GlobalAveragePooling1D, \
    MaxPool2D, Concatenate, Add, Flatten, Conv2D, Input, Dense, \
    Dropout, Activation, BatchNormalization
import keras.backend as K


def inception_block(inputs, filters):
    """
    inception 架构模块
    Args:
        inputs: 输入数据的尺寸
        filters: 核数量

    Returns:

    """
    # 分支1
    conv_1 = Conv1D(filters=filters, kernel_size=1, strides=1, padding="same", activation="relu")(inputs)

    # 分支2
    conv_2 = Conv1D(filters=filters, kernel_size=3, strides=1, padding="same", activation="relu")(inputs)

    # 分支3
    conv_3 = Conv1D(filters=filters, kernel_size=3, strides=1, padding="same", activation="relu")(inputs)
    conv_3 = Conv1D(filters=filters, kernel_size=3, strides=1, padding="same", activation="relu")(conv_3)

    # 合并
    outputs = Concatenate(axis=-1)([conv_1, conv_2, conv_3])
    outputs = Conv1D(filters=filters, kernel_size=1, strides=1, padding="same", activation="relu")(outputs)

    return outputs


def se_block(inputs, k):  # SE Block模块
    # 输入尺寸
    # ip = Input(shape=(data_lenght,))
    input_shape = K.int_shape(inputs)
    # input_shape = Input(shape=(200,))

    # 全局平均池化
    outputs = GlobalAveragePooling1D()(inputs)

    # 计算每个通道的重要性
    outputs = Dense(units=int(input_shape[-1] / k), activation="relu")(outputs)
    outputs = Dense(units=input_shape[-1], activation="sigmoid")(outputs)

    # 重新标定每个通道
    outputs = RepeatVector(input_shape[1])(outputs)
    outputs = Reshape([input_shape[1], input_shape[2]])(outputs)
    outputs = Multiply()([inputs, outputs])

    return outputs


# inception maxpooling selection layer
def ims_layer(inputs, filters, pool_size):
    """
    特征提取层
    Args:
        inputs: 输入数据的尺寸
        filters: 核数量
        pool_size: 池化步长

    Returns:

    """

    inception = inception_block(inputs, filters)
    pool = MaxPool1D(pool_size=pool_size, strides=pool_size, padding="same")(inception)
    se = se_block(pool, 4)

    return se


def fc_layer(inputs, units):
    """
    最后的全连接层
    Args:
        inputs:
        units:

    Returns:

    """
    outputs = Dense(units=units, activation="relu")(inputs)
    outputs = Dropout(0.5)(outputs)

    return outputs

from keras.layers import Reshape
# 这个方法可以引用一篇 inception+se 的文章
def inception_model(no_activities=7, data_lenght=2000, kernel_number_base=64, kernel_wide_base=1):
    """
    原始输入数据
    Args:
        data_length:

    Returns:

    """
    # raw_input = Input((data_length, 1))

    print('no_activities: %d\n' % (no_activities))
    # ip = Input(shape=(data_lenght,))
    # raw_input = Embedding(128,
    #                 64,
    #                 # weights=[embedding_matrix],
    #                 input_length=data_lenght,
    #                 trainable=True)(ip)

    ip = Input((data_lenght,))

    raw_input = Reshape((-1, 1), input_shape=(data_lenght, ))(ip)

    # ims_1
    ims_1 = ims_layer(raw_input, kernel_number_base*1, 2)

    # ims_2
    ims_2 = ims_layer(ims_1, kernel_number_base*2, 3)

    # ims_3
    ims_3 = ims_layer(ims_2, kernel_number_base*4, 3)

    # ims_4
    ims_4 = ims_layer(ims_3, kernel_number_base*4, 3)

    # Flatten
    flatten = Flatten()(ims_4)

    # 连接原始信号特征和相关系数
    # concat

    # fc
    # fc_1
    fc_1 = fc_layer(flatten, kernel_number_base*8)
    # fc_2
    fc_2 = fc_layer(fc_1, kernel_number_base*8)

    # x_output
    x_output = Dense(units=no_activities, activation="softmax")(fc_2)

    # 建立模型
    model = Model(inputs=ip, outputs=x_output)

    return model


if __name__ == '__main__':
    from keras.utils.vis_utils import plot_model

    kernel_wide_base = 8
    kernel_number_base = 8
    net_deep_base = 2

    model = WCNNR(no_activities=7, kernel_wide_base=kernel_wide_base, kernel_number_base=kernel_number_base,net_deep_base=net_deep_base)
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    print("success")
    pass