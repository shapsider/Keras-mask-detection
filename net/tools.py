"""必要的工具"""
import numpy as np
import tensorflow as tf
from PIL import Image
import keras
import numpy as np
import math

def get_new_img_size(width, height, img_min_side=600):
    """把图片最小边resize到600"""
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height


class BBoxUtility(object):
    def __init__(self, priors=None, overlap_threshold=0.7, ignore_threshold=0.3,
                 nms_thresh=0.7, top_k=300):
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self.ignore_threshold = ignore_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k
        self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        self.scores = tf.placeholder(dtype='float32', shape=(None,))
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

    @property
    def nms_thresh(self):
        return self._nms_thresh

    @nms_thresh.setter
    def nms_thresh(self, value):
        self._nms_thresh = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, value):
        self._top_k = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)

    def iou(self, box):
        """
        box：真实框
        计算真实框与所有先验框的IOU值
        """
        # 计算出每个真实框与所有的先验框的iou
        # 判断真实框与先验框的重合情况
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # 真实框的面积
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # 先验框的面积
        area_gt = (self.priors[:, 2] - self.priors[:, 0]) * (self.priors[:, 3] - self.priors[:, 1])
        # 计算iou
        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True):
        """如果重合度超过0.7，认为可以进行调节成真实框"""
        #box为传入的真实框，计算与先验框的重合度
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_priors, 4 + return_iou))

        # 找到每一个真实框，重合程度较高的先验框
        # 如果iou>0.7，认为可以利用这个先验框回归到真实框
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]

        #对真实框与先验框进行编码，计算出应该有的预测结果，用于网络回归训练
        # 找到对应的先验框
        assigned_priors = self.priors[assign_mask]
        # 逆向编码，将真实框转化为FasterRCNN预测结果的格式
        # 先计算真实框的中心与长宽
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        # 再计算重合度较高的先验框的中心与长宽
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])

        # 逆向求取FasterRCNN应该有的预测结果
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        encoded_box[:, :2][assign_mask] *= 4

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
        encoded_box[:, 2:4][assign_mask] *= 4
        return encoded_box.ravel()

    def ignore_box(self, box):
        """
        box:标记中真实的框
        """
        # 获取所有先验框和真实框的重合程度
        iou = self.iou(box)

        ignored_box = np.zeros((self.num_priors, 1))

        # 找到每一个真实框，重合程度较高的先验框
        # 如果重合程度大于0.3或者小于0.7，就应该忽略这个框
        assign_mask = (iou > self.ignore_threshold) & (iou < self.overlap_threshold)

        if not assign_mask.any():
            assign_mask[iou.argmax()] = True

        #找出需要忽略的先验框
        ignored_box[:, 0][assign_mask] = iou[assign_mask]
        return ignored_box.ravel()

    def assign_boxes(self, boxes, anchors):
        """计算真实框对应的先验框，与这个先验框应当有的预测结果"""
        #先求出先验框个数
        self.num_priors = len(anchors)
        self.priors = anchors
        #创建一个全0矩阵，第一维是先验框个数，第二维的前4列是先验框的位置信息，第5列代表是否包含物体
        assignment = np.zeros((self.num_priors, 4 + 1))

        #初始时，认为所有先验框为背景，所以第5列=0
        assignment[:, 4] = 0.0
        if len(boxes) == 0:
            return assignment

        # 对每一个真实框都进行iou计算，找到需要忽略的先验框
        ingored_boxes = np.apply_along_axis(self.ignore_box, 1, boxes[:, :4])
        # 取重合程度最大的先验框，并且获取这个先验框的index
        ingored_boxes = ingored_boxes.reshape(-1, self.num_priors, 1)
        # (num_priors)
        ignore_iou = ingored_boxes[:, :, 0].max(axis=0)
        # (num_priors)
        ignore_iou_mask = ignore_iou > 0

        #代表忽略此先验框，将ignore_iou_mask的序号下的先验框设为忽略
        assignment[:, 4][ignore_iou_mask] = -1

        # (n, num_priors, 5)encode_box将真实框进行编码，计算出训练使用的预测结果：即偏移信息
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        # 每一个真实框的编码后的值，和iou
        # (n, num_priors)
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)

        #由于某些先验框会与多个真实框重合，所以要找iou最大的真实框进行对应
        # 取重合程度最大的先验框，并且获取这个先验框的index
        # (num_priors)
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        # (num_priors)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        # (num_priors)
        best_iou_mask = best_iou > 0
        # 某个先验框它属于哪个真实框
        best_iou_idx = best_iou_idx[best_iou_mask]

        assign_num = len(best_iou_idx)
        # 保留重合程度最大的先验框的应该有的预测结果
        # 哪些先验框存在真实框
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]

        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        # 4代表为背景的概率，为0
        # 1为正样本，代表有物体，0为负样本，代表背景，-1代表不要的框
        assignment[:, 4][best_iou_mask] = 1
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的
        # assignment即为找到的合理先验框
        return assignment

    def decode_boxes(self, mbox_loc, mbox_priorbox):
        """
        使用边框回归公式得到调整后的框
        mbox_loc: RPN输出的先验框的调整参数
        mbox_priorbox: 传入根据特征图生成好的先验框
        """
        # 获得先验框的宽与高
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]

        # 获得先验框的中心点
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

        # 真实框距离先验框中心的xy轴偏移情况
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width / 4
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_height / 4
        decode_bbox_center_y += prior_center_y

        # 真实框的宽与高的求取
        decode_bbox_width = np.exp(mbox_loc[:, 2] / 4)
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] / 4)
        decode_bbox_height *= prior_height

        # 获取真实框的左上角与右下角
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        # 真实框的左上角与右下角进行堆叠
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        # 为方便计算，框的坐标值设置在0-1，即防止超出0与1
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out(self, predictions, mbox_priorbox, num_classes, keep_top_k=300,confidence_threshold=0.5):
        """
        初步选出300个建议框
        predictions：RPN输出的结果，对每个先验框都要计算
        mbox_priorbox：传入根据特征图生成好的先验框
        """

        # 网络预测的结果
        # x_class即置信度
        mbox_conf = predictions[0]
        # 先验框的调整参数
        mbox_loc = predictions[1]
        # 先验框
        mbox_priorbox = mbox_priorbox
        results = []
        # 对每一个框进行处理
        for i in range(len(mbox_loc)):
            results.append([])
            #得到调整后的框，这里是所有的框
            decode_bbox = self.decode_boxes(mbox_loc[i], mbox_priorbox)
            #选出含物体的框，并用NMS删除重叠框
            for c in range(num_classes):
                c_confs = mbox_conf[i, :, c]
                #先进行对比，如果x_class的概率大于设置的置信度，认为框里有物体
                c_confs_m = c_confs > confidence_threshold
                if len(c_confs[c_confs_m]) > 0:
                    # 取出得分高于confidence_threshold的框
                    boxes_to_process = decode_bbox[c_confs_m]
                    # 获取我们所选出的框的置信度用于NMS
                    confs_to_process = c_confs[c_confs_m]
                    # 进行iou的非极大抑制，删除重叠框
                    feed_dict = {self.boxes: boxes_to_process,
                                 self.scores: confs_to_process}
                    idx = self.sess.run(self.nms, feed_dict=feed_dict)
                    # 取出在非极大抑制中效果较好的内容
                    good_boxes = boxes_to_process[idx]
                    confs = confs_to_process[idx][:, None]
                    # 将label、置信度、框的位置进行堆叠，label代表是否有物体，作用其实不大，因为置信度也可以反映
                    labels = c * np.ones((len(idx), 1))
                    c_pred = np.concatenate((labels, confs, good_boxes),
                                            axis=1)
                    # 添加进result里
                    results[-1].extend(c_pred)
            #将以上得到的框排序，选出前k个作为RPN的输出框，默认k=300
            if len(results[-1]) > 0:
                # 按照置信度进行排序
                results[-1] = np.array(results[-1])
                argsort = np.argsort(results[-1][:, 1])[::-1]
                results[-1] = results[-1][argsort]
                # 选出置信度最大的keep_top_k个
                results[-1] = results[-1][:keep_top_k]
        # 获得，在所有预测结果里面，置信度比较高的框
        # 另外，利用先验框和RPN的预测结果x_class,x_regr，调整到真实框（预测框）的位置
        return results

    def nms_for_out(self, all_labels, all_confs, all_bboxes, num_classes, nms):
        results = []
        nms_out = tf.image.non_max_suppression(self.boxes, self.scores,
                                               self._top_k,
                                               iou_threshold=nms)
        for c in range(num_classes):
            c_pred = []
            mask = all_labels == c
            if len(all_confs[mask]) > 0:
                # 取出得分高于confidence_threshold的框
                boxes_to_process = all_bboxes[mask]
                confs_to_process = all_confs[mask]
                # 进行iou的非极大抑制
                feed_dict = {self.boxes: boxes_to_process,
                             self.scores: confs_to_process}
                idx = self.sess.run(nms_out, feed_dict=feed_dict)
                # 取出在非极大抑制中效果较好的内容
                good_boxes = boxes_to_process[idx]
                confs = confs_to_process[idx][:, None]
                # 将label、置信度、框的位置进行堆叠。
                labels = c * np.ones((len(idx), 1))
                c_pred = np.concatenate((labels, confs, good_boxes), axis=1)
            results.extend(c_pred)
        return results