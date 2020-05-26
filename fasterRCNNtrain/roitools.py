"""ROI计算工具"""
import numpy as np
import copy

def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def iou(a, b):
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def calc_iou(R, config, all_boxes, width, height, num_classes):
    # print(all_boxes)
    bboxes = all_boxes[:, :4]
    gta = np.zeros((len(bboxes), 4))

    #将真实框的格式映射到特征层大小上
    for bbox_num, bbox in enumerate(bboxes):
        gta[bbox_num, 0] = int(round(bbox[0] * width / config.rpn_stride))
        gta[bbox_num, 1] = int(round(bbox[1] * height / config.rpn_stride))
        gta[bbox_num, 2] = int(round(bbox[2] * width / config.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox[3] * height / config.rpn_stride))
    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = []
    # print(gta)

    #将建议框的格式映射到特征层大小上
    for ix in range(R.shape[0]):
        x1 = R[ix, 0] * width / config.rpn_stride
        y1 = R[ix, 1] * height / config.rpn_stride
        x2 = R[ix, 2] * width / config.rpn_stride
        y2 = R[ix, 3] * height / config.rpn_stride

        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))
        # print([x1, y1, x2, y2])
        best_iou = 0.0
        best_bbox = -1

        #计算每一个建议框与所有真实框的重合程度
        for bbox_num in range(len(bboxes)):
            curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 1], gta[bbox_num, 2], gta[bbox_num, 3]], [x1, y1, x2, y2])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num
        # print(best_iou)

        #iou过小直接跳过这个建议框
        if best_iou < config.classifier_min_overlap:
            continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])
            IoUs.append(best_iou)

            # iou大小适中，当成负样本，意思是可以回归，但是不能线性回归
            if config.classifier_min_overlap <= best_iou < config.classifier_max_overlap:
                label = -1

            # 如果建议框的iou大于设置阈值0.5，就认为这个建议框可以线性回归到真实框
            elif config.classifier_max_overlap <= best_iou:

                #对建议框和真实框进行编码
                label = int(all_boxes[best_bbox, -1])
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 2]) / 2.0
                cyg = (gta[best_bbox, 1] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 2] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 1]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError
        # print(label)
        class_label = num_classes * [0]
        class_label[label] = 1
        y_class_num.append(copy.deepcopy(class_label))
        coords = [0] * 4 * (num_classes - 1)
        labels = [0] * 4 * (num_classes - 1)
        if label != -1:
            label_pos = 4 * label
            sx, sy, sw, sh = config.classifier_regr_std
            coords[label_pos:4 + label_pos] = [sx * tx, sy * ty, sw * tw, sh * th]
            labels[label_pos:4 + label_pos] = [1, 1, 1, 1]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None, None

    # 将可以线性回归的建议框转变成训练需要的形式，以便计算损失进行训练
    # X代表经过再次处理后的建议框，这些建议框与特征层对应(包含正负样本，正样本可线性回归，负样本不能线性回归)
    # Y1为分类信息
    # Y2为回归信息
    X = np.array(x_roi)
    # print(X)
    Y1 = np.array(y_class_num)
    Y2 = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)], axis=1)

    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs