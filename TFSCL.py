import keras_metrics as km
import matplotlib.pyplot as plt
from keras.optimizers import adam_v2
import tensorflow as tf
from keras import backend as K
from keras import layers as KL
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, Dense, Dropout, add,Lambda,GlobalAveragePooling1D,\
    BatchNormalization, AveragePooling1D, Activation, Flatten,Input,concatenate,LayerNormalization
from tensorflow.keras.utils import plot_model
from keras.models import Model
from keras.regularizers import l2

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
sess=tf.compat.v1.Session(config=config)

num_classes = 10

def FFTLayer(x):
    x_comp = tf.cast(x, tf.complex64)
    x_fft = tf.signal.fft(x_comp)
    x_abs = tf.abs(x_fft)
    return x_abs



def Final_loss(y_true, y_pred,f1,f2):
    loss=K.mean(CLoss(f1, f2) + K.binary_crossentropy(y_true,y_pred))
    return loss


def Scos(f1, f2):
    f1 = tf.math.l2_normalize(f1, axis=1)
    f2 = tf.math.l2_normalize(f2, axis=1)
    cos=tf.reduce_mean(tf.reduce_sum((f1 * f2), axis=1))
    return (1-cos)

class CL_Loss(KL.Layer):
    def __init__(self, **kwargs):
        super(CL_Loss, self).__init__(**kwargs)
    def call(self, inputs, **kwargs):

        f1,f2 = inputs
        loss = K.mean(Scos(f1, f2))

        self.add_loss(loss, inputs=True )
        self.add_metric(loss, aggregation="mean", name="CL_loss")
        return loss

def cnn_model(filters, kernerl_size, strides, conv_padding, dil_rate, inputs):
    x = Conv1D(filters=16, kernel_size=3, strides=1,
               padding='same', kernel_regularizer=l2(1e-4),activation=tf.nn.gelu)(inputs)
    x = Conv1D(filters=filters, kernel_size=kernerl_size, strides=strides,
               padding=conv_padding, dilation_rate=dil_rate, kernel_regularizer=l2(1e-4),activation=tf.nn.gelu)(x)
    return x


def time_brach(inputs,BatchNormal=True):
    modelA = cnn_model(filters=16, kernerl_size=3, strides=1, conv_padding='same', dil_rate=1, inputs=inputs)
    modelB = cnn_model(filters=16, kernerl_size=3, strides=1, conv_padding='same', dil_rate=2, inputs=inputs)
    modelC = cnn_model(filters=16, kernerl_size=3, strides=1, conv_padding='same', dil_rate=3, inputs=inputs)
    combined = concatenate([modelA, modelB, modelC])
    x =  Conv1D(filters=32, kernel_size=3, strides=1,
               padding='same',  kernel_regularizer=l2(1e-4),activation='relu')(combined)
    if BatchNormal:
        x = BatchNormalization()(x)
    x =  Conv1D(filters=64, kernel_size=3, strides=1,
               padding='same',  kernel_regularizer=l2(1e-4),activation='relu')(x)
    if BatchNormal:
        x = BatchNormalization()(x)
    return x

def freq_brach(inputs):
    x = Lambda(FFTLayer)(inputs)
    x = Conv1D(filters=16, kernel_size=3, strides=1,
               padding='same', kernel_regularizer=l2(1e-4),activation=tf.nn.gelu)(x)
    x = Conv1D(filters=32, kernel_size=3, strides=1,
               padding='same', kernel_regularizer=l2(1e-4),activation=tf.nn.gelu)(x)
    x = Conv1D(filters=32, kernel_size=3, strides=1,
               padding='same', kernel_regularizer=l2(1e-4),activation=tf.nn.gelu)(x)
    x = Conv1D(filters=64, kernel_size=3, strides=1,
               padding='same', kernel_regularizer=l2(1e-4),activation=tf.nn.gelu)(x)

    return x


def TFSCL(x_shape=(5120,10)):
    inputs_x = Input(x_shape, name='x_train')

    fea1 = freq_brach(inputs_x)
    fea2 = time_brach(inputs_x)

    res_t=Conv1D(filters=64, kernel_size=1, strides=1,
               padding='same', kernel_regularizer=l2(1e-4),activation=tf.nn.gelu)(inputs_x)


    res_f= Lambda(FFTLayer)(inputs_x)
    res_f=Conv1D(filters=64, kernel_size=1, strides=1,
               padding='same', kernel_regularizer=l2(1e-4),activation=tf.nn.gelu)(res_f)


    fea1 = add([fea1,res_t])
    fea2 = add([fea2,res_f])
    loss = CL_Loss()([fea1, fea2])
    con=concatenate([fea1,fea2])+loss
    con=GlobalAveragePooling1D()(con)


    y_pred = Dense(units=num_classes, activation='sigmoid', kernel_regularizer=l2(1e-4),name='output')(con)



    model = Model(inputs=inputs_x, outputs=y_pred)
    adam = adam_v2.Adam(learning_rate=0.001)
    model.compile(optimizer=adam)

    model.summary()
    plot_model(model=model, to_file='TFSCL_model.png', show_shapes=True)
    return model



def Train_Eval():
    #path = r""  #your data path
    #x_train, y_train, x_valid, y_valid, x_test, y_test =  # Read the dataset through your own functions

    x_shape = x_train.shape[1:]

    model = TFSCL(x_shape=x_shape)
    model.summary()
    callback_list = [ModelCheckpoint(filepath='propose.hdf5', verbose=1, save_best_only=True, monitor="loss")]
    model.fit([x_train,y_train],batch_size=64, epochs=20,shuffle=True,verbose=1, callbacks=callback_list)
    model.load_weights('propose.hdf5')

    # 搭建评估模型

    pred_model = Model(inputs=model.get_layer('x_train').input,outputs=model.get_layer('output').output)

    adam = adam_v2.Adam(learning_rate=0.001)
    pred_model.compile(optimizer=adam, loss='binary_crossentropy',
                  metrics='acc')

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss:", loss)
    print("test accuracy", accuracy)



if __name__ == "__main__":
    Model=TFSCL()



