"""
训练需要的txt文件
根据train.txt生成year_train.txt
第一列：图片存放的绝对路径
第二列：坐标
第三联：对应的类别 后面的列依次也是：坐标 类别
"""
import xml.etree.ElementTree as ET
from os import getcwd

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

wd = getcwd()
#classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
#classes改成自己要的类
classes =["face","face_mask"]

def convert_annotation(year, image_id, list_file):
    # xml标记的路径
    in_file = open(r'D:\Python37\User\envs\MASK\maskProject\ObjectDetect\fasterRCNN\myfrcnn\MyVOC2007\Annotations\%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    if root.find('object')==None:
        return
    # 图片路径
    list_file.write(r'D:\Python37\User\envs\MASK\maskProject\ObjectDetect\fasterRCNN\myfrcnn\MyVOC2007\JPEGImages\%s.jpg'%(image_id))
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    list_file.write('\n')

for year, image_set in sets:
    # 训练的索引txt文件路径
    image_ids = open(r'D:\Python37\User\envs\MASK\maskProject\ObjectDetect\fasterRCNN\myfrcnn\MyVOC2007\ImageSets\Main\%s.txt'%(image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        convert_annotation(year, image_id, list_file)
    list_file.close()
