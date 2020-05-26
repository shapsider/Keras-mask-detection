"""最后的分类输出层"""
from keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D,TimeDistributed,Add
from keras.layers import Activation,Flatten,BatchNormalization
from keras import backend as K
from net import roipooling as roipooling

def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):
    """
    与backbone中的block不同之处在于，此处需要对每一批即32个池化后的区域(14,14,1024)进行残差卷积
    TimeDistributed的作用是将一批分解成时间序列分别处理，输入输出是一批一批的
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 在每个操作前用TimeDistributed实现分批处理
    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2c')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x

def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c', trainable=trainable)(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    shortcut = TimeDistributed(Conv2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def classifier_layers(x, input_shape, trainable=False):
    """分类层本质是resnet50的最后一个stage"""
    # 输入是（Batch_size，32个建议框，14,14,1024），两次identity_block_td后得到（Batch_size，32个建议框，7，7，2048）
    x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2, 2), trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)#（Batch_size，32个建议框，7，7，2048）

    # 池化后为（Batch_size，32个建议框，1，1，2048）
    x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)
    return x

def end_classify(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    """与前面的roipooling进行结合得到一个完整的接口"""
    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)

    #pooling_regions, num_rois为层的初始化定义参数，pooling_regions是池化后输出的固定大小14*14
    #base_layers[38,38,1024]
    #input_rois为建议框[None,4]，None与每次处理的建议框数量num_rois有关，config中定义为32
    out_roi_pool =roipooling.RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    #roipooling后进行分类
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
    # 展开成一个固定长度的向量，注意是每批32个向量
    out = TimeDistributed(Flatten())(out)
    # 全连接nb_classes个数的输出代表类别(包含背景)，解释一下，每批32个，即此处为（32,nb_classes）,nb_classes中只有一个有效
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # 全连接（nb_classes-1）*4 个数的输出代表框的进一步调整，只调整（nb_classes-1）中概率最大的那个
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    # 从classifier_layers中输出的是（Batch_size，32个建议框，1，1，2048），如果nb_classes=21
    # out_class为（Batch_size，32个建议框，21）
    # out_regr为（Batch_size，32个建议框，80）
    return [out_class, out_regr]
