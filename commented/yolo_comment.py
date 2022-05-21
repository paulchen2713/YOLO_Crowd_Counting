#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys # conversions between color systems
import os
from timeit import default_timer as timer #return the value of a counter,can measure a short duration

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw #PIL:python imaging library

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image #resize image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 使用第一張GPU卡
from keras.utils import multi_gpu_model
gpu_num=1

class YOLO(object):
    def __init__(self):
        self.model_path = 'model_data/yolo.h5' # model path or trained weights path
        self.anchors_path = 'model_data/yolo_anchors.txt' # 錨點 anchor point ?
        self.classes_path = 'model_data/coco_classes.txt' # 分類項目
        self.score = 0.3
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors() 
        self.sess = K.get_session() #獲取keras已建立的session
        self.model_image_size = (416, 416) # fixed size or (None, None), hw #輸入尺寸
        self.boxes, self.scores, self.classes = self.generate() #由generate()完成目標檢測

    #透過讀取類別文件獲取類別名
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names] #默認刪除字符串頭和尾的空格和換行符
        return class_names
    
    #讀取anchor文件中的數據
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2) #行數未知，列數=2

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.' #判斷model是否以.h5結尾

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors) #num_anchors=9
        num_classes = len(self.class_names) #num_classes=80
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.) #色調,飽和度,亮度
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)) #hsv轉RGB
        self.colors = list( 
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), # hsv:0~1，RBG:0~255
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, )) #K.placeholder:分配空間。張量有一維度，長度=2

        #若gpu數大於2，調用multi_gpu_model()
        if gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer() #定時器
        
        #圖片的寬、高必須調整成32的倍數
        if self.model_image_size != (None, None): #判斷圖片是否存在
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size))) #進行圖片縮放
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape) #(416，416,3)
        image_data /= 255. #將縮放後的圖片數值除以255，做規格化
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.#批量增加一維

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]], #圖片尺寸:416x416
                K.learning_phase(): 0 # 0:測試模式 / 1:訓練模式
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img')) #繪製邊框

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32')) #設置字體
        thickness = (image.size[0] + image.size[1]) // 300 #框線的寬度

        #對於c個目標目標類別中的每個目標框i，使用pillow繪圖
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image) #output original image
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            #top和left的座標小數點後一位四捨五入
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            #bottom和right的座標小數點後一位四捨五入，與圖片相比，取最小值
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32')) 
            print(label, (left, top), (right, bottom))

            #確定lable起始點左、下
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle( #畫框
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle( #文字背景
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font) #標籤內容
            del draw

        end = timer() #結束計時
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():#確認影片是否存在
        raise IOError("Couldn't open webcam or video")
    #獲得影片屬性、fps、size
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0 #current fps
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()

        image = Image.fromarray(frame) #array轉成image
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        #圖片加上文字
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

#開啟目標圖片
def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()



if __name__ == '__main__':
    detect_img(YOLO())
