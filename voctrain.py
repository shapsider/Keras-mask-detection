"""基于VOC格式的数据集训练"""
from fasterRCNNtrain import RCNNtrain as RCNNtrain

if __name__ == "__main__":
    annotation_path=r"D:\Python37\User\envs\MASK\maskProject\ObjectDetect\fasterRCNN\myfrcnn\voctools\2007_train.txt"
    base_net_weights = r"D:\Python37\User\envs\MASK\maskProject\ObjectDetect\fasterRCNN\myfrcnn\model_data\pertrain_weights.h5"
    #.h5会保存到logs日志下
    log_dir=r"D:\Python37\User\envs\MASK\maskProject\ObjectDetect\fasterRCNN\myfrcnn\model_data\logs"
    RCNNtrain.train_faster(annotation_path, base_net_weights, log_dir)