import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Reshape, Flatten, Input, Conv2D, Concatenate, Lambda, BatchNormalization, ReLU, Conv1D, DepthwiseConv1D


def convert_candel_to_1d_array(candels_reshape, filters):

    conv2d_1 = Conv2D(filters=filters, kernel_size=(4, 3),
                      strides=(2, 1))(candels_reshape)
    conv2d_2 = Conv2D(filters=filters, kernel_size=(8, 3),
                      strides=(2, 1))(candels_reshape)
    conv2d_3 = Conv2D(filters=filters, kernel_size=(
        16, 3), strides=(2, 1))(candels_reshape)
    conv2d_4 = Conv2D(filters=filters, kernel_size=(
        32, 3), strides=(2, 1))(candels_reshape)
    conv2d_5 = Conv2D(filters=filters, kernel_size=(
        64, 3), strides=(2, 1))(candels_reshape)
    conv2d_6 = Conv2D(filters=filters, kernel_size=(
        128, 3), strides=(2, 1))(candels_reshape)

    batch_norm_1 = BatchNormalization()(conv2d_1)
    batch_norm_2 = BatchNormalization()(conv2d_2)
    batch_norm_3 = BatchNormalization()(conv2d_3)
    batch_norm_4 = BatchNormalization()(conv2d_4)
    batch_norm_5 = BatchNormalization()(conv2d_5)
    batch_norm_6 = BatchNormalization()(conv2d_6)

    relu_1 = ReLU()(batch_norm_1)
    relu_2 = ReLU()(batch_norm_2)
    relu_3 = ReLU()(batch_norm_3)
    relu_4 = ReLU()(batch_norm_4)
    relu_5 = ReLU()(batch_norm_5)
    relu_6 = ReLU()(batch_norm_6)

    reshape_1 = Reshape(target_shape=(
        relu_1.shape[1], relu_1.shape[3]))(relu_1)
    reshape_2 = Reshape(target_shape=(
        relu_2.shape[1], relu_2.shape[3]))(relu_2)
    reshape_3 = Reshape(target_shape=(
        relu_3.shape[1], relu_3.shape[3]))(relu_3)
    reshape_4 = Reshape(target_shape=(
        relu_4.shape[1], relu_4.shape[3]))(relu_4)
    reshape_5 = Reshape(target_shape=(
        relu_5.shape[1], relu_5.shape[3]))(relu_5)
    reshape_6 = Reshape(target_shape=(
        relu_6.shape[1], relu_6.shape[3]))(relu_6)

    return reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6


def one_stride_conv_block(reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, filters):

    depthwiseconv1d_1 = DepthwiseConv1D(
        kernel_size=4, padding='same')(reshape_1)
    depthwiseconv1d_2 = DepthwiseConv1D(
        kernel_size=8, padding='same')(reshape_2)
    depthwiseconv1d_3 = DepthwiseConv1D(
        kernel_size=16, padding='same')(reshape_3)
    depthwiseconv1d_4 = DepthwiseConv1D(
        kernel_size=32, padding='same')(reshape_4)
    depthwiseconv1d_5 = DepthwiseConv1D(
        kernel_size=64, padding='same')(reshape_5)
    depthwiseconv1d_6 = DepthwiseConv1D(
        kernel_size=128, padding='same')(reshape_6)

    batch_norm_1 = BatchNormalization()(depthwiseconv1d_1)
    batch_norm_2 = BatchNormalization()(depthwiseconv1d_2)
    batch_norm_3 = BatchNormalization()(depthwiseconv1d_3)
    batch_norm_4 = BatchNormalization()(depthwiseconv1d_4)
    batch_norm_5 = BatchNormalization()(depthwiseconv1d_5)
    batch_norm_6 = BatchNormalization()(depthwiseconv1d_6)

    relu_1 = ReLU()(batch_norm_1)
    relu_2 = ReLU()(batch_norm_2)
    relu_3 = ReLU()(batch_norm_3)
    relu_4 = ReLU()(batch_norm_4)
    relu_5 = ReLU()(batch_norm_5)
    relu_6 = ReLU()(batch_norm_6)

    conv1d_1 = Conv1D(filters=filters, kernel_size=4, padding='same')(relu_1)
    conv1d_2 = Conv1D(filters=filters, kernel_size=8, padding='same')(relu_2)
    conv1d_3 = Conv1D(filters=filters, kernel_size=16, padding='same')(relu_3)
    conv1d_4 = Conv1D(filters=filters, kernel_size=32, padding='same')(relu_4)
    conv1d_5 = Conv1D(filters=filters, kernel_size=64, padding='same')(relu_5)
    conv1d_6 = Conv1D(filters=filters, kernel_size=128, padding='same')(relu_6)

    batch_norm_1 = BatchNormalization()(conv1d_1)
    batch_norm_2 = BatchNormalization()(conv1d_2)
    batch_norm_3 = BatchNormalization()(conv1d_3)
    batch_norm_4 = BatchNormalization()(conv1d_4)
    batch_norm_5 = BatchNormalization()(conv1d_5)
    batch_norm_6 = BatchNormalization()(conv1d_6)

    relu_1 = ReLU()(batch_norm_1)
    relu_2 = ReLU()(batch_norm_2)
    relu_3 = ReLU()(batch_norm_3)
    relu_4 = ReLU()(batch_norm_4)
    relu_5 = ReLU()(batch_norm_5)
    relu_6 = ReLU()(batch_norm_6)

    return relu_1, relu_2, relu_3, relu_4, relu_5, relu_6


def two_stride_conv_block(relu_1, relu_2, relu_3, relu_4, relu_5, relu_6, filters):

    depthwiseconv1d_1 = DepthwiseConv1D(
        kernel_size=4, padding='same', strides=2)(relu_1)
    depthwiseconv1d_2 = DepthwiseConv1D(
        kernel_size=8, padding='same', strides=2)(relu_2)
    depthwiseconv1d_3 = DepthwiseConv1D(
        kernel_size=16, padding='same', strides=2)(relu_3)
    depthwiseconv1d_4 = DepthwiseConv1D(
        kernel_size=32, padding='same', strides=2)(relu_4)
    depthwiseconv1d_5 = DepthwiseConv1D(
        kernel_size=64, padding='same', strides=2)(relu_5)
    depthwiseconv1d_6 = DepthwiseConv1D(
        kernel_size=128, padding='same', strides=2)(relu_6)

    batch_norm_1 = BatchNormalization()(depthwiseconv1d_1)
    batch_norm_2 = BatchNormalization()(depthwiseconv1d_2)
    batch_norm_3 = BatchNormalization()(depthwiseconv1d_3)
    batch_norm_4 = BatchNormalization()(depthwiseconv1d_4)
    batch_norm_5 = BatchNormalization()(depthwiseconv1d_5)
    batch_norm_6 = BatchNormalization()(depthwiseconv1d_6)

    relu_1 = ReLU()(batch_norm_1)
    relu_2 = ReLU()(batch_norm_2)
    relu_3 = ReLU()(batch_norm_3)
    relu_4 = ReLU()(batch_norm_4)
    relu_5 = ReLU()(batch_norm_5)
    relu_6 = ReLU()(batch_norm_6)

    conv1d_1 = Conv1D(filters=filters, kernel_size=4, padding='same')(relu_1)
    conv1d_2 = Conv1D(filters=filters, kernel_size=8, padding='same')(relu_2)
    conv1d_3 = Conv1D(filters=filters, kernel_size=16, padding='same')(relu_3)
    conv1d_4 = Conv1D(filters=filters, kernel_size=32, padding='same')(relu_4)
    conv1d_5 = Conv1D(filters=filters, kernel_size=64, padding='same')(relu_5)
    conv1d_6 = Conv1D(filters=filters, kernel_size=128, padding='same')(relu_6)

    batch_norm_1 = BatchNormalization()(conv1d_1)
    batch_norm_2 = BatchNormalization()(conv1d_2)
    batch_norm_3 = BatchNormalization()(conv1d_3)
    batch_norm_4 = BatchNormalization()(conv1d_4)
    batch_norm_5 = BatchNormalization()(conv1d_5)
    batch_norm_6 = BatchNormalization()(conv1d_6)

    relu_1 = ReLU()(batch_norm_1)
    relu_2 = ReLU()(batch_norm_2)
    relu_3 = ReLU()(batch_norm_3)
    relu_4 = ReLU()(batch_norm_4)
    relu_5 = ReLU()(batch_norm_5)
    relu_6 = ReLU()(batch_norm_6)

    return relu_1, relu_2, relu_3, relu_4, relu_5, relu_6


input_layer = Input(shape=(256, 3))
reshaped_candels = Reshape(target_shape=(256, 3, 1))(input_layer)
# Converting candels into 1d arrays

reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6 = convert_candel_to_1d_array(
    candels_reshape=reshaped_candels,
    filters=32
)


# Block1 with strides = 1, shape sizes of: 127 , 125 , 121 , 113 , 97 , 65

relu_1, relu_2, relu_3, relu_4, relu_5, relu_6 = one_stride_conv_block(
    reshape_1,
    reshape_2,
    reshape_3,
    reshape_4,
    reshape_5,
    reshape_6,
    filters=32
)


# Block 2 with stride = 2, shape sizes of: 64 , 63 , 61 , 57 , 49 , 33
relu_1, relu_2, relu_3, relu_4, relu_5, relu_6 = two_stride_conv_block(
    relu_1,
    relu_2,
    relu_3,
    relu_4,
    relu_5,
    relu_6,
    filters=64
)

# Block 3 with stride = 2, shape sizes of: 32 , 32, 31, 29, 25 , 17
relu_1, relu_2, relu_3, relu_4, relu_5, relu_6 = two_stride_conv_block(
    relu_1,
    relu_2,
    relu_3,
    relu_4,
    relu_5,
    relu_6,
    filters=128
)

# Block 4 with stride = 2, shape sizes of: 16, 16, 16, 15, 13,  9
relu_1, relu_2, relu_3, relu_4, relu_5, relu_6 = two_stride_conv_block(
    relu_1,
    relu_2,
    relu_3,
    relu_4,
    relu_5,
    relu_6,
    filters=256
)

# Block 5 with stride = 2, shape sizes of: 8, 8, 8, 8, 7, 5
relu_1, relu_2, relu_3, relu_4, relu_5, relu_6 = two_stride_conv_block(
    relu_1,
    relu_2,
    relu_3,
    relu_4,
    relu_5,
    relu_6,
    filters=256
)


global_avg_pool_1 = keras.layers.GlobalAveragePooling1D()(relu_1)
global_avg_pool_2 = keras.layers.GlobalAveragePooling1D()(relu_2)
global_avg_pool_3 = keras.layers.GlobalAveragePooling1D()(relu_3)
global_avg_pool_4 = keras.layers.GlobalAveragePooling1D()(relu_4)
global_avg_pool_5 = keras.layers.GlobalAveragePooling1D()(relu_5)
global_avg_pool_6 = keras.layers.GlobalAveragePooling1D()(relu_6)


# print(f'{relu_1.shape=}')
# print(f'{relu_2.shape=}')
# print(f'{relu_3.shape=}')
# print(f'{relu_4.shape=}')
# print(f'{relu_5.shape=}')
# print(f'{relu_6.shape=}')

model = keras.Model(
    inputs=input_layer,

    outputs=[
        global_avg_pool_1,
        global_avg_pool_2,
        global_avg_pool_3,
        global_avg_pool_4,
        global_avg_pool_5,
        global_avg_pool_6
    ]
)

print(model.summary())
