#!/usr/bin/env python
# encoding: utf-8
"""
@file: people_flow.py

"""

import colorsys
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.utils import multi_gpu_model


class YOLO(object):
    def __init__(self): # YOLO參數
        self.model_path = 'model_data/yolo.h5' # 已訓練完成的模型，支持重新訓練的模型
        self.anchors_path = 'model_data/yolo_anchors.txt' # anchor box的配置文件，9個寬高組合,錨點框
        self.classes_path = 'model_data/coco_classes.txt' # 類別文件,，與模型文件匹配
        self.score = 0.3 # 置信度的閾值，刪除小於閾值的候選框
        self.iou = 0.45 # 候選框的IoU閾值，刪除同類別中大於閾值的候選框
        self.class_names = self._get_class()#類別列表，讀取classes_path
        self.anchors = self._get_anchors()#anchor box列表，讀取anchors_path
        self.sess = K.get_session()
        self.model_image_size = (416, 416) # 模型所檢測圖像的尺寸，輸入圖像都需要按此填充
        self.boxes, self.scores, self.classes = self.generate() # 檢測的核心輸出，函數generate()所生成，是模型的輸出封裝

    def _get_class(self): # 載入類別文件
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self): # 載入\anchor box組合文件
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path) #取得model位置
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'#判斷model是否為.h5檔

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors) # anchor box的總數
        num_classes = len(self.class_names) # 類別總數
        is_tiny_version = num_anchors==6 # default setting
        try:
            # the loaded model is assigned to "model.yolo_model"
            self.yolo_model = load_model(model_path, compile=False) # load the model
        except:
             self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \  #由yolo_body所創建的模型
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes):
             self.yolo_model.load_weights(self.model_path)   # make sure model, anchors and classes match,# 加載模型參數

        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.生成繪製邊界框的顏色
        hsv_tuples = [(x / len(self.class_names), 1., 1.) # 色調h:(x / len(self.class_names),飽和度s:1.0,明亮v:1.0
                      
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)) #hsv色域轉換為rgb色域
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors)) #hsv變為rgb的值轉換,hsv取值範圍在【0,1】，而RBG取值範圍在【0,255】，所以乘上255
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)#yolo_eval:yolo評估函數
        
        return boxes, scores, classes#return框的座標boxes，框的類別置信度scores，框的類別classes

    def detect_image(self, image):
        start = timer()
#檢測方法第1步:圖像處理
        if self.model_image_size != (None, None): #判斷圖片是否存在
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'#斷言圖片的寬是32的倍數
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'#斷言圖片的高是32的倍數
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),#調整圖片寬為32的倍數 
                              image.height - (image.height % 32))#調整圖片高為32的倍數
            boxed_image = letterbox_image(image, new_image_size) #填充圖像
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
#第2步:餵入數據，圖像，圖像尺寸
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],#求boxes,scores,classes,由generate()計算
            feed_dict={#喂參數
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0 #學習模式0:測試模型,1:訓練模式
            })
        lbox = []
        lscore = []
        lclass = []
        for i in range(len(out_classes)): 
            if out_classes[i] == 0:
                lbox.append(out_boxes[i])#將out_boxes內容填入1box陣列?
                lscore.append(out_scores[i])#將out_scores內容填入1score陣列?
                lclass.append(out_classes[i])#將out_classes內容填入1class陣列?
        out_boxes = np.array(lbox)
        out_scores = np.array(lscore)
        out_classes = np.array(lclass)
        print('There are {} people in this image.'.format(len(out_boxes)))#
#第3步:繪製邊框，自動設置邊框寬度，繪製邊框和類別文字，使用Pillow繪圖庫
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))#字體
        thickness = (image.size[0] + image.size[1]) // 300 #厚度

        font_cn = ImageFont.truetype(font='font/asl.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)#標籤文字

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))#輸出邊界框


            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):#畫出邊界框
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle( # 文字背景
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            show_str = 'There are'+str(len(out_boxes))+'people in this image.'
            label_size1 = draw.textsize(show_str, font_cn)
            print(label_size1)
            draw.rectangle(
                [10, 10, 10 + label_size1[0], 10 + label_size1[1]],
                fill=(255,255,0))
            draw.text((10,10),show_str,fill=(0, 0, 0), font=font_cn)

            del draw

        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()

        image = Image.fromarray(frame)
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
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()


def detect_img(yolo):#使用已經訓練完成的YOLO v3模型，檢測圖片中的物體
    while True:
        img = input('Input image filename:')#輸入一張圖片
        try:
            image = Image.open(img) #使用Image.open()加載圖像
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image) #調用yolo.detect_image()檢測圖像
            r_image.show()#顯示檢測完成的圖像r_image
    yolo.close_session()#關閉yolo的session



if __name__ == '__main__':
    detect_img(YOLO())
