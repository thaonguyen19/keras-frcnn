from __future__ import print_function
from __future__ import absolute_import

from keras.applications.mobilenet import MobileNet
from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, \
    AveragePooling2D, TimeDistributed, Dropout

from keras import backend as K
from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras_frcnn.FixedBatchNormalization import FixedBatchNormalization


def get_weight_path():
    return 'mobilenet-weights-best.hdf5'

def get_img_output_length(width, height):
    def get_output_length(input_length):
        filter_sizes = [3, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1]#, 3, 1, 3, 1]
        strides = [2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]#, 2, 1, 1, 1]
        assert len(filter_sizes) == len(strides)
        for i in range(len(filter_sizes)):
            input_length = (input_length + strides[i] - 1) // strides[i]
        return input_length
    return get_output_length(width), get_output_length(height) 

def nn_base(input_tensor=None, trainable=False):
    assert K.image_dim_ordering() == 'tf'
    input_shape = (None, None, 3)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    base_model = MobileNet(input_shape=None, include_top=False, weights='imagenet')
    print ("training with IMAGENET weights")
    ################### REMOVE THE LAST 2 BLOCKS (ids 12 and 13) #####################
    for i in range(12):
        base_model.layers.pop()
    base_model.outputs = [base_model.layers[-1].output]
    base_model.layers[-1].outbound_nodes = []
    return base_model(img_input)

def rpn(base_layers,num_anchors):
    x = Convolution2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]

# def classifier_layers(x, input_shape, trainable=False):
#     assert K.backend() == 'tensorflow'
#     x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2, 2), trainable=trainable)

#     x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
#     x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
#     x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)

#     return x

def classifier(base_layers, input_rois, num_rois, nb_classes = 21, trainable=False):
    assert K.backend() == 'tensorflow'
    pooling_regions = 14
    input_shape = (num_rois,14,14,1024)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]



