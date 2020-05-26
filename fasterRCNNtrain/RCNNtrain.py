"""
网络训练
首先，正负样本分成3个阶段
第一阶段是RPN的先验框是否包含物体（包含具体物体为正样本，包含背景为负样本），这可以初步预测出建议框
第二阶段是判断建议框与真实框的重合度，（可以线性回归的是正样本，必须非线性回归的是负样本）
第三阶段是最终选出的建议框中，所含物体的类别，多个二分类构成的多分类
其次，训练分两个阶段
第一阶段，训练RPN得到建议框
第二阶段，使用RPN提取建议框，训练建议框中的物体分类与边框细致回归
"""
import sys
sys.path.append("..")

from net import fasterrcnn as frcnn
from fasterRCNNtrain import loss_and_gen as loss_and_gen

from net import netconfig as netconfig
from net import tools as tools
from fasterRCNNtrain import roitools as roitools

from keras.utils import generic_utils
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import keras
import numpy as np
import time
import tensorflow as tf
from net import RPN as RPN


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def train_faster(annotation_path,base_net_weights,log_dir):
    # 构建网络需要的参数
    config = netconfig.Config()
    NUM_CLASSES = 3
    EPOCH = 16
    # 每一代训练的次数
    EPOCH_LENGTH = 1000
    # 帮助编码和解码，训练rpn时应将框编码成GT可以匹配的格式
    bbox_util = tools.BBoxUtility(overlap_threshold=config.rpn_max_overlap, ignore_threshold=config.rpn_min_overlap)
    # 为了方便训练，将标注的xml提取到txt中
    #annotation_path =r"D:\Python37\User\envs\MASK\maskProject\ObjectDetect\fasterRCNN\myfrcnn\voctools\2007_train.txt"

    # 构建两个网络：RPN和最后用于的分类的网络
    model_rpn, model_classifier, model_all = frcnn.get_model(config, NUM_CLASSES)
    # 预训练权重
    #base_net_weights =r"D:\Python37\User\envs\MASK\maskProject\ObjectDetect\fasterRCNN\myfrcnn\model_data\pertrain_weights.h5"

    model_all.summary()
    model_rpn.load_weights(base_net_weights, by_name=True)
    model_classifier.load_weights(base_net_weights, by_name=True)

    # 从txt训练数据中读取每一行，并打乱
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 数据很多，不能一次性fit，所以需要使用数据生成器,solid=True,训练图片大小强制resize
    gen = loss_and_gen.Generator(bbox_util, lines, NUM_CLASSES, solid=True)
    rpn_train = gen.generate()

    #log_dir = "logs"

    # 训练参数设置，加载日志
    logging = TensorBoard(log_dir=log_dir)
    callback = logging
    callback.set_model(model_all)

    # 因为有两个网络，需要两次编译
    model_rpn.compile(loss={'regression': loss_and_gen.smooth_l1(), 'classification': loss_and_gen.cls_loss()},
                      optimizer=keras.optimizers.Adam(lr=1e-4))
    model_classifier.compile(loss=[loss_and_gen.class_loss_cls, loss_and_gen.class_loss_regr(NUM_CLASSES - 1)],
                             metrics={'dense_class_{}'.format(NUM_CLASSES): 'accuracy'},
                             optimizer=keras.optimizers.Adam(lr=1e-4))
    model_all.compile(optimizer='sgd', loss='mae')

    # 初始化参数，用于控制台打印训练信息
    iter_num = 0
    train_step = 0
    losses = np.zeros((EPOCH_LENGTH, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()
    # 最佳loss
    best_loss = np.Inf
    # 数字到类的映射
    print('Starting training')

    for i in range(EPOCH):

        # 当到达20代时，减小学习率，epoch修改实验
        if i == 4:
            model_rpn.compile(loss={'regression': loss_and_gen.smooth_l1(), 'classification': loss_and_gen.cls_loss()},
                              optimizer=keras.optimizers.Adam(lr=1e-5))
            model_classifier.compile(loss=[loss_and_gen.class_loss_cls, loss_and_gen.class_loss_regr(NUM_CLASSES - 1)],
                                     metrics={'dense_class_{}'.format(NUM_CLASSES): 'accuracy'},
                                     optimizer=keras.optimizers.Adam(lr=1e-5))
            print("Learning rate decrease")

        progbar = generic_utils.Progbar(EPOCH_LENGTH)
        print('Epoch {}/{}'.format(i + 1, EPOCH))
        while True:
            if len(rpn_accuracy_rpn_monitor) == EPOCH_LENGTH and config.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                    mean_overlapping_bboxes, EPOCH_LENGTH))
                if mean_overlapping_bboxes == 0:
                    print(
                        'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

            # 使用next就能不停从生成器获取训练数据
            # X为预处理后的图片，Y为训练需要的边框调整信息，boxes为真实框
            X, Y, boxes = next(rpn_train)

            loss_rpn = model_rpn.train_on_batch(X, Y)
            write_log(callback, ['rpn_cls_loss', 'rpn_reg_loss'], loss_rpn, train_step)

            # 使用训练好的RPN获得建议框
            P_rpn = model_rpn.predict_on_batch(X)
            height, width, _ = np.shape(X[0])
            anchors = RPN.create_anchor(loss_and_gen.get_img_output_length(width, height), width, height)

            # 将预测结果进行解码获得建议框在原图上的位置
            results = bbox_util.detection_out(P_rpn, anchors, 1, confidence_threshold=0)

            # R对应原图的建议框
            R = results[0][:, 2:]

            # 使用建议框计算与真实框之间的重合程度，求出分类层边框第二次回归训练需要的信息
            # X2代表经过再次处理后的建议框，这个建议框与特征层对应(包含正负样本，正样本可线性回归，负样本不能线性回归)
            # Y1为分类信息
            # Y2为回归信息
            X2, Y1, Y2, IouS = roitools.calc_iou(R, config, boxes[0], width, height, NUM_CLASSES)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            # 以下是进行正负样本的平衡
            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if len(neg_samples) == 0:
                continue

            if len(pos_samples) < config.num_rois // 2:
                selected_pos_samples = pos_samples.tolist()
            else:
                selected_pos_samples = np.random.choice(pos_samples, config.num_rois // 2, replace=False).tolist()
            try:
                selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples),
                                                        replace=False).tolist()
            except:
                selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples),
                                                        replace=True).tolist()

            sel_samples = selected_pos_samples + selected_neg_samples

            # 将正负样本传入训练
            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                         [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            write_log(callback, ['detection_cls_loss', 'detection_reg_loss', 'detection_acc'], loss_class, train_step)

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]
            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            train_step += 1
            iter_num += 1
            progbar.update(iter_num,
                           [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                            ('detector_cls', np.mean(losses[:iter_num, 2])),
                            ('detector_regr', np.mean(losses[:iter_num, 3]))])

            if iter_num == EPOCH_LENGTH:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if config.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                        mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                write_log(callback,
                          ['Elapsed_time', 'mean_overlapping_bboxes', 'mean_rpn_cls_loss', 'mean_rpn_reg_loss',
                           'mean_detection_cls_loss', 'mean_detection_reg_loss', 'mean_detection_acc', 'total_loss'],
                          [time.time() - start_time, mean_overlapping_bboxes, loss_rpn_cls, loss_rpn_regr,
                           loss_class_cls, loss_class_regr, class_acc, curr_loss], i)

                if config.verbose:
                    print('The best loss is {}. The current loss is {}. Saving weights'.format(best_loss, curr_loss))
                if curr_loss < best_loss:
                    best_loss = curr_loss
                model_all.save_weights(log_dir + "/epoch{:03d}-loss{:.3f}-rpn{:.3f}-roi{:.3f}".format(i, curr_loss,
                                                                                                      loss_rpn_cls + loss_rpn_regr,
                                                                                                      loss_class_cls + loss_class_regr) + ".h5")

                break