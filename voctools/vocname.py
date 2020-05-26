"""批量生成VOC文件名"""
import os

#格式JPEGImages
path = r"D:/Python37/User/envs/MASK/maskProject/ObjectDetect/fasterRCNN/myfrcnn/MyVOC2007/JPEGImages/" # 目标路径

"""os.listdir(path) 操作效果为 返回指定路径(path)文件夹中所有文件名"""
filename_list = os.listdir(path)  # 扫描目标路径的文件,将文件名存入列表

stk_code = 0
for i in filename_list:
    used_name = path + filename_list[stk_code]
    stk_code_str = str(stk_code).zfill(6)
    #print(stk_code_str)
    new_name = path + stk_code_str +".jpg"
    os.rename(used_name,new_name)
    #print("文件%s重命名成功,新的文件名为%s" %(used_name,new_name))
    stk_code += 1

#格式Annotations
path = r"D:/Python37/User/envs/MASK/maskProject/ObjectDetect/fasterRCNN/myfrcnn/MyVOC2007/Annotations/"
filename_list = os.listdir(path)
stk_code = 0
for i in filename_list:
    used_name = path + filename_list[stk_code]
    stk_code_str = str(stk_code).zfill(6)
    new_name = path + stk_code_str +".xml"
    os.rename(used_name,new_name)
    stk_code += 1

