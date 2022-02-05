#!/usr/bin/env python3

from keras.layers import Dense, LSTM, Bidirectional, Average, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.initializers import he_uniform

from keras import regularizers



from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, MaxPool1D, GlobalAveragePooling1D, \
    concatenate, Activation
from keras.models import Model



# input_dim: int > 0。
# output_dim: int >= 0。
def get_LSTM(input_dim, output_dim, max_lenght, no_activities):
    model = Sequential(name='LSTM')
    model.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))  # output_dim * input_dim
    model.add(LSTM(output_dim))  # (output_dim + input_dim + 1) * （output_dim * 4）

    model.add(Dense(no_activities, activation='softmax'))  # (input_dim * no_activities + no_activites)
    return model


def WCNN(no_activities):
    multiply = 8
    print('no_activities: %d\n' % (no_activities))
    ip = Input(shape=(2000,))
    emb = Embedding(188,
                    64,
                    # weights=[embedding_matrix],
                    input_length=2000,
                    trainable=True)(ip)

    emb = BatchNormalization()(emb)

    ly_he_uniform_1 = he_uniform(1)
    cnn1 = Conv1D(4*multiply, 1, padding='same', kernel_initializer=ly_he_uniform_1)(emb)

    cnn1 = Activation('relu')(cnn1)


    ly_he_uniform_2 = he_uniform(2)
    cnn2 = Conv1D(4*multiply, 3, padding='same', kernel_initializer=ly_he_uniform_2)(emb)
    cnn2 = Activation('relu')(cnn2)


    ly_he_uniform_3 = he_uniform(3)
    cnn3 = Conv1D(2*multiply, 5, padding='same', kernel_initializer=ly_he_uniform_3)(emb)
    cnn3 = Activation('relu')(cnn3)

    ly_he_uniform_4 = he_uniform(4)
    cnn4 = Conv1D(1 * multiply, 7, padding='same', kernel_initializer=ly_he_uniform_4)(emb)
    cnn4 = Activation('relu')(cnn4)

    cnn = concatenate([cnn1, cnn2, cnn3, cnn4])

    cnn = GlobalMaxPooling1D()(cnn)
    cnn = BatchNormalization(axis=1)(cnn)

    dcnn1 = Dense(4*multiply)(cnn)
    dcnn1 = Activation('relu')(dcnn1)
    dcnn1 = Dropout(0.4)(dcnn1)


    dcnn2 = Dense(4*multiply)(dcnn1)
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
