"""Resnet50前4个stage"""
from keras import layers
from keras.layers import Input
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers import Activation,BatchNormalization

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """改变特征层的维度"""
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    #第一次卷积
    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    #第二次卷积
    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    # 第三次卷积
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    #残差边的卷积
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)
    #两边相加
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """加深网络深度"""
    #最后一个滤波器的数量应该等于输入张量的维数
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    #此处体现了与conv_block的不同之处
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def ResNet50(inputs):

    img_input = inputs#假设(600,600,3)

    x = ZeroPadding2D((3, 3))(img_input)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)#(300,300,64)
    x = BatchNormalization(name='bn_conv1')(x)#标准化
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)#(150,150,64)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))#150,150,256
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')#150,150,256
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')#150,150,256

    #150*150->75*75是因为默认设置了strides=(2,2)
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')#75,75,512
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')#75,75,512
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')#75,75,512
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')#75,75,512

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')#38,38,1024
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')#38,38,1024
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')#38,38,1024
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')#38,38,1024
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')#38,38,1024
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')#38,38,1024
    #输出公共特征层38,38,1024

    """
    全连接层之前应该还有一个stage5，fasterrcnn中stage5被用到图像分类任务
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    """

    #model = Model(img_input, x, name='resnet50')#目前该语句是为了可以加载模型打印摘要

    return x

if __name__ == "__main__":
    inputs = Input(shape=(600, 600, 3))
    #Tensor("input_1:0", shape=(?, 600, 600, 3), dtype=float32)
    print(inputs)
    model=ResNet50(inputs) 
    #model,model_summary= ResNet50(inputs)
    #model_summary.summary()