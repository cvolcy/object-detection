import numpy as np
import onnxruntime as rt
import colorsys
from PIL import Image,ImageDraw

numClasses = 20 # Number of classes | Based on YOLO model

classColor = [colorsys.hsv_to_rgb(x*1.0/numClasses, 0.5, 0.5) for x in range(numClasses)]
classColor = [(round(i[0] * 255), round(i[1] * 255), round(i[2] * 255)) for i in classColor]

label = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
        "cow","diningtable","dog","horse","motorbike","person","pottedplant",
        "sheep","sofa","train","tvmonitor"]

labelIndex = np.random.randint(0, numClasses, size=1)[0]

print(f"Class : {label[labelIndex]}, Color : {classColor[labelIndex]}")