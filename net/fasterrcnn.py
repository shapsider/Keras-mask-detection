"""fasterRCNN构建，这里只是返回模型"""
from net import backbone as backbone
from net import RPN as RPN
from net import classify as classify
from keras.layers import Input
from keras.models import Model

def get_model(flag,class_num):
    """直观反映流程，用于训练"""
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    #共享特征层
    base_layers = backbone.ResNet50(inputs)
    #默认为9
    anchor_num = len(flag.anchor_box_scales) * len(flag.anchor_box_ratios)
    #建立RPN
    rpn = RPN.RPN(base_layers,anchor_num)
    #[:2]切片，只选择列表内最初的两个索引0,1
    #index=0，N行1列9*坐标个数的概率值，index=1，N行4列，N行1列9*坐标个数的的偏移信息
    model_rpn = Model(inputs, rpn[:2])

    #roi_input为建议框[None, 4]，None与每次处理的建议框数量num_rois有关，config中定义为32
    # classifier为[out_class,out_regr]
    # out_class为（Batch_size，32个建议框，21）
    # out_regr为（Batch_size，32个建议框，80）
    classifier = classify.end_classify(base_layers, roi_input, flag.num_rois, nb_classes=class_num, trainable=True)
    model_classifier = Model([inputs, roi_input], classifier)

    #model_all实际上是合并了RPN网络和分类网络,即为fasterrcnn网络
    model_all = Model([inputs, roi_input], rpn[:2]+classifier)
    return model_rpn,model_classifier,model_all

def get_predict_model(config,num_classes):
    """用于预测"""
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    feature_map_input = Input(shape=(None,None,1024))

    base_layers = backbone.ResNet50(inputs)
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = RPN.RPN(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn)

    classifier = classify.end_classify(feature_map_input, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    return model_rpn,model_classifier_only


