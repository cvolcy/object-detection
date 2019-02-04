import numpy as np
import onnxruntime as rt
import colorsys
from PIL import Image,ImageDraw

IMG_PATH = 'tests/test1.jpg'
ANCHORS = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

numClasses = 20 # Number of classes | Based on YOLO model

classColor = [colorsys.hsv_to_rgb(x*1.0/numClasses, 0.5, 0.5) for x in range(numClasses)]
classColor = [(round(i[0] * 255), round(i[1] * 255), round(i[2] * 255)) for i in classColor]

label = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
        "cow","diningtable","dog","horse","motorbike","person","pottedplant",
        "sheep","sofa","train","tvmonitor"]

sess = rt.InferenceSession("models/tiny_yolov2.onnx")

input_names = list(map(lambda x: x.name,sess.get_inputs()))
input_name = input_names[0]
print(f"Input names : {input_names}")
print(f"Output names : {list(map(lambda x: x.name,sess.get_outputs()))}\n")


print(f"Load image {IMG_PATH}")
img = Image.open(IMG_PATH)
original_size = img.size
YOLO_SIZE = (416, 416)
print(f"Resizing image from {original_size} to {YOLO_SIZE}")
ow, oh = img.size
rw = ow/YOLO_SIZE[0]
rh = oh/YOLO_SIZE[1]
X = np.asarray(img.resize(YOLO_SIZE))
X = X.transpose(2,0,1)
X = X.reshape(1,3,YOLO_SIZE[0],YOLO_SIZE[1])

print(f"Test image shape{X.shape}")

out = sess.run(None, {input_name: X.astype(np.float32)})
out = out[0][0]

def sigmoid(x):
    return 1/(1+np.exp(-x))
def softmax(x):
    scoreMatExp = np.exp(np.asarray(x))
    return scoreMatExp / scoreMatExp.sum(0)

draw = ImageDraw.Draw(img)
found = 0
for cy in range(0,13):
    for cx in range(0,13):
        for b in range(0,5):
            channel = b*(numClasses+5)
            tx = out[channel  ][cy][cx]
            ty = out[channel+1][cy][cx]
            tw = out[channel+2][cy][cx]
            th = out[channel+3][cy][cx]
            tc = out[channel+4][cy][cx]
            x = (float(cx) + sigmoid(tx))*32
            y = (float(cy) + sigmoid(ty))*32

            w = np.exp(tw) * 32 * ANCHORS[2*b  ]
            h = np.exp(th) * 32 * ANCHORS[2*b+1] 

            confidence = sigmoid(tc)

            classes = np.zeros(numClasses)
            for c in range(0,numClasses):
                classes[c] = out[channel + 5 +c][cy][cx]

            classes = softmax(classes)
            detectedClass = classes.argmax()

            if 0.6 < classes[detectedClass] * confidence and confidence > 0.4:
                found = found + 1
                print(classes[detectedClass] * confidence, label[detectedClass]+str(found), confidence, classes[detectedClass])