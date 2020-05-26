"""共享特征层后的RPN"""
import keras
from keras.layers import Conv2D,Input,TimeDistributed,Flatten,Dense,Reshape
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from net import netconfig as netconfig

#基本设置
config = netconfig.Config()

def RPN(baselayers,anchor_num):
    """RPN初步提取出建议"""
    #3*3卷积，生成通道数为512的特征层
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(baselayers)

    #两个分支，class分支与理论有所不同，不是18输出通道，是9输出
    x_class = Conv2D(anchor_num, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    #回归分支每个坐标输出36个通道
    x_regr = Conv2D(anchor_num * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    #特征平铺(-1,1)行数未知，列数为1，靠近1代表有物体的可能性大，靠近0反之，(-1,4)行数未知，列数为4
    #x_class反映每个框是否包含物体，x_regr反映边框修正的偏移信息
    x_class = Reshape((-1, 1), name="classification")(x_class)
    x_regr = Reshape((-1, 4), name="regression")(x_regr)
    return [x_class, x_regr, baselayers]

"""先验框生成，每个点对应的9个框"""

def generate_anchors(sizes=None, ratios=None):
    """生成9个框"""
    if sizes is None:
        sizes = config.anchor_box_scales
    if ratios is None:
        ratios = config.anchor_box_ratios
    num_anchors = len(sizes) * len(ratios)
    anchors = np.zeros((num_anchors, 4))
    anchors[:, 2:] = np.tile(sizes, (2, len(ratios))).T
    for i in range(len(ratios)):
        anchors[3 * i:3 * i + 3, 2] = anchors[3 * i:3 * i + 3, 2] * ratios[i][0]
        anchors[3 * i:3 * i + 3, 3] = anchors[3 * i:3 * i + 3, 3] * ratios[i][1]
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    #返回9行4列矩阵，代表设置好的9个框
    return anchors

def shift(shape, anchors, stride=config.rpn_stride):
    #shape指特征图的形状，stride为经过backbone后缩放的比例，600*600->38*38，stride=16
    #shift_x和shift_y代表特征图映射到原图的每个网格的中心
    # [0,1,2,3,4,5……37]
    # [0.5,1.5,2.5……37.5]
    # 比如原图x方向上的中心：[8,24,……]
    shift_x = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride
    shift_y = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])
    #以上获得了特征图到原图的所有中心点
    #print(shift_x,shift_y)
    #[  8.  24.  40. ... 568. 584. 600.] [  8.   8.   8. ... 600. 600. 600.]

    #在行上堆叠
    shifts = np.stack([shift_x,shift_y,shift_x,shift_y], axis=0)

    shifts = np.transpose(shifts)
    number_of_anchors = np.shape(anchors)[0]#9个

    k = np.shape(shifts)[0]

    #使用shifts后与anchors中的框信息相加得到先验框左上角和右下角坐标
    shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]),keras.backend.floatx())
    #reshape到最后一个维度为4，即4列，分别是左上角坐标，右下角坐标，k * number_of_anchors为所有先验框个数
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])

    """
    #先验框的可视化
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(-300, 900)
    plt.xlim(-300, 900)
    # plt.ylim(0,600)
    # plt.xlim(0,600)
    plt.scatter(shift_x, shift_y)
    box_widths = shifted_anchors[:, 2] - shifted_anchors[:, 0]
    box_heights = shifted_anchors[:, 3] - shifted_anchors[:, 1]
    #选择特征图上的某个点，比如选择中间点，+shape[0]/2*9是因为从0开始计数
    initial = int(shape[0]*shape[1]/2*9+shape[0]/2*9)
    print(shape[0])#特征图的x轴，x1,x2,x3...
    print(shape[1])#特征图的y轴，y1,y2,y3...
    print(initial)
    for i in [initial + 0, initial + 1, initial + 2, initial + 3, initial + 4, initial + 5, initial + 6, initial + 7, initial + 8]:
        rect = plt.Rectangle([shifted_anchors[i, 0], shifted_anchors[i, 1]], box_widths[i], box_heights[i], color="r", fill=False)
        ax.add_patch(rect)
    plt.show()
    """

    return shifted_anchors

def create_anchor(shape,width,height):
    """shape是特征图的尺寸，width和height是原图的尺寸"""
    #产生9个先验框偏移信息
    anchors = generate_anchors()
    #根据输入的特征图产生出原图上所有先验框左上角坐标和右下角坐标
    network_anchors = shift(shape,anchors)
    #为了加快计算，转换成小数，转换到值域为0-1内
    network_anchors[:,0] = network_anchors[:,0]/width
    network_anchors[:,1] = network_anchors[:,1]/height
    network_anchors[:,2] = network_anchors[:,2]/width
    network_anchors[:,3] = network_anchors[:,3]/height
    #删除超出原图范围的先验框
    network_anchors = np.clip(network_anchors,0,1)

    return network_anchors

if __name__ == "__main__":
    #anchor=generate_anchors()
    """得到9个框
    [[ -64.  -64.   64.   64.]
     [-128. -128.  128.  128.]
     [-256. -256.  256.  256.]
     [ -64. -128.   64.  128.]
     [-128. -256.  128.  256.]
     [-256. -512.  256.  512.]
     [-128.  -64.  128.   64.]
     [-256. -128.  256.  128.]
     [-512. -256.  512.  256.]]
    """
    #print(anchor)
    #shift([38,38],anchor)
    net_anchor=create_anchor([38,38],600,600)
    """0-1内的左上角坐标和右下角坐标
    [[0.         0.         0.12       0.12      ]
    [0.         0.         0.22666667 0.22666667]
    [0.         0.         0.44       0.44      ]
    ...
    [0.78666667 0.89333333 1.         1.        ]
    [0.57333333 0.78666667 1.         1.        ]
    [0.14666667 0.57333333 1.         1.        ]]
    """
    print(net_anchor)