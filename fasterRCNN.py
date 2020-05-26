"""fasterRCNN对象创建"""
import numpy as np
import colorsys
import os
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image, ImageFont, ImageDraw
import copy
import math

from net import fasterrcnn as frcnn
from net import netconfig as netconfig
from net import RPN as RPN
from net import tools as tools

class FasterRCNN(object):

    _defaults = {
        "model_path": './model_data/logs/epoch015-loss1.729-rpn1.025-roi0.704.h5',
        "classes_path": './model_data/index.txt',
        "confidence": 0.7,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        """初始化faster RCNN"""
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.sess = K.get_session()
        self.config = netconfig.Config()
        self.generate()
        self.bbox_util = tools.BBoxUtility()
        self.confidence = 0.7
        self.classes_path='./model_data/index.txt'
        self.model_path='./model_data/logs/epoch015-loss1.729-rpn1.025-roi0.704.h5'

    def _get_class(self):
        """获得所有的分类"""
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def generate(self):
        """获得所有的分类"""
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # 计算总的种类
        self.num_classes = len(self.class_names) + 1

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        self.model_rpn, self.model_classifier = frcnn.get_predict_model(self.config, self.num_classes)
        self.model_rpn.load_weights(self.model_path, by_name=True)
        self.model_classifier.load_weights(self.model_path, by_name=True, skip_mismatch=True)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    def get_img_output_length(self, width, height):
        def get_output_length(input_length):
            # input_length += 6
            filter_sizes = [7, 3, 1, 1]
            padding = [3, 1, 0, 0]
            stride = 2
            for i in range(4):
                # input_length = (input_length - filter_size + stride) // stride
                input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
            return input_length

        return get_output_length(width), get_output_length(height)

    def detect_image(self, image):
        """检测图片"""
        image_shape = np.array(np.shape(image)[0:2])
        old_width = image_shape[1]
        old_height = image_shape[0]
        # 保存原始图片
        old_image = copy.deepcopy(image)

        # 把图片的最短边resize到600
        width, height = tools.get_new_img_size(old_width, old_height)
        image = image.resize([width, height])
        # 图片转成数组
        photo = np.array(image, dtype=np.float64)

        # 图片预处理，归一化
        photo = preprocess_input(np.expand_dims(photo, 0))
        # 使用RPN预测，获得概率x_class和x_regr
        preds = self.model_rpn.predict(photo)

        # 将预测结果进行解码
        # 获得所有先验框
        anchors = RPN.create_anchor(self.get_img_output_length(width, height), width, height)

        # 解码获得建议框，这里得到了300个建议框，注意其坐标均为0-1间
        rpn_results = self.bbox_util.detection_out(preds, anchors, 1, confidence_threshold=0)

        # 将返回的0-1的建议框映射到共享特征图，如果特征图为38*38，值域变成0-38之间，R为300行4列，分别是左上角右下角坐标
        R = rpn_results[0][:, 2:]
        R[:, 0] = np.array(np.round(R[:, 0] * width / self.config.rpn_stride), dtype=np.int32)
        R[:, 1] = np.array(np.round(R[:, 1] * height / self.config.rpn_stride), dtype=np.int32)
        R[:, 2] = np.array(np.round(R[:, 2] * width / self.config.rpn_stride), dtype=np.int32)
        R[:, 3] = np.array(np.round(R[:, 3] * height / self.config.rpn_stride), dtype=np.int32)
        print(R)

        # R转换一下，前两列是左上角坐标，后两列是宽和高
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
        base_layer = preds[2]

        delete_line = []
        for i, r in enumerate(R):
            if r[2] < 1 or r[3] < 1:
                delete_line.append(i)
        R = np.delete(R, delete_line, axis=0)

        bboxes = []
        probs = []
        labels = []

        # 分批次遍历建议框，每批32个
        for jk in range(R.shape[0] // self.config.num_rois + 1):
            # 取出32个建议框
            ROIs = np.expand_dims(R[self.config.num_rois * jk:self.config.num_rois * (jk + 1), :], axis=0)

            # 判断建议框是否有效
            if ROIs.shape[1] == 0:
                break

            # 对最后一次整除不全，不能到32个的建议框小批进行填充
            if jk == R.shape[0] // self.config.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], self.config.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            # 将共享特征层和建议框传入end_classifier进行预测
            # P_cls为（Batch_size，32个建议框，21）
            # P_regr为（Batch_size，32个建议框，80）
            [P_cls, P_regr] = self.model_classifier.predict([base_layer, ROIs])

            # 判断输出的每批中每个建议框是否真实包含我们要的物体，本身置信度阈值设置为0.9，如果是背景也要跳过
            for ii in range(P_cls.shape[1]):
                # P_cls[0, ii, :-1]是21个概率组成的列表
                if np.max(P_cls[0, ii, :-1]) < self.confidence or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                # 获得label
                label = np.argmax(P_cls[0, ii, :-1])
                # 获得坐标信息
                (x, y, w, h) = ROIs[0, ii, :]

                # 其实就是label
                cls_num = np.argmax(P_cls[0, ii, :-1])

                # 获取框的信息，并改变数量级
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= self.config.classifier_regr_std[0]
                ty /= self.config.classifier_regr_std[1]
                tw /= self.config.classifier_regr_std[2]
                th /= self.config.classifier_regr_std[3]

                # 获取到共享特征层上真实的坐标信息
                cx = x + w / 2.
                cy = y + h / 2.
                cx1 = tx * w + cx
                cy1 = ty * h + cy
                w1 = math.exp(tw) * w
                h1 = math.exp(th) * h

                x1 = cx1 - w1 / 2.
                y1 = cy1 - h1 / 2.

                x2 = cx1 + w1 / 2
                y2 = cy1 + h1 / 2

                x1 = int(round(x1))
                y1 = int(round(y1))
                x2 = int(round(x2))
                y2 = int(round(y2))

                # bboxes是最终从300个建议框过滤出来与目标物体对应的建议框
                # 但注意，这里的建议框还是存在重叠现象，因为之前仅仅靠物体置信度来筛选
                bboxes.append([x1, y1, x2, y2])
                probs.append(np.max(P_cls[0, ii, :-1]))
                labels.append(label)

        # 没检测到物体，返回
        if len(bboxes) == 0:
            return old_image

        # 将38*38特征层的建议框映射到600*600
        # 筛选出其中得分高于confidence的框，因此此时需要再次NMS删除重叠框
        labels = np.array(labels)
        probs = np.array(probs)
        boxes = np.array(bboxes, dtype=np.float32)
        boxes[:, 0] = boxes[:, 0] * self.config.rpn_stride / width
        boxes[:, 1] = boxes[:, 1] * self.config.rpn_stride / height
        boxes[:, 2] = boxes[:, 2] * self.config.rpn_stride / width
        boxes[:, 3] = boxes[:, 3] * self.config.rpn_stride / height
        results = np.array(
            self.bbox_util.nms_for_out(np.array(labels), np.array(probs), np.array(boxes), self.num_classes - 1, 0.4))

        top_label_indices = results[:, 0]
        top_conf = results[:, 1]

        boxes = results[:, 2:]
        #top_label_indices=labels
        #top_conf=probs

        # 大小调整到原图上，此时已经完成了建议框的计算
        boxes[:, 0] = boxes[:, 0] * old_width
        boxes[:, 1] = boxes[:, 1] * old_height
        boxes[:, 2] = boxes[:, 2] * old_width
        boxes[:, 3] = boxes[:, 3] * old_height

        # simhei.ttf用于设置字体
        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = (np.shape(old_image)[0] + np.shape(old_image)[1]) // old_width * 2
        image = old_image
        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = top_conf[i]

            left, top, right, bottom = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def close(self):
        self.sess.close()