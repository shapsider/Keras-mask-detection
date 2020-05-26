"""使用fasterRCNN进行目标检测"""
from fasterRCNN import FasterRCNN
from PIL import Image

if __name__ == "__main__":
    fasterrcnn = FasterRCNN()
    img = "./valimg/mask2.jpg"
    image = Image.open(img)
    pred_image = fasterrcnn.detect_image(image)
    pred_image.show()
    fasterrcnn.close()