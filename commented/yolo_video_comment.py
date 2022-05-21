import sys

if len(sys.argv) < 2: #從 command line 傳入的參數 < 2，不帶參數
    print("Usage: $ python {0} [video_path] [output_path(optional)]", sys.argv[0]) #sys.argv[0]:程式名稱
    exit() #退出

from yolo import YOLO
from yolo import detect_video

if __name__ == '__main__':
    video_path = sys.argv[1] #參數1
    if len(sys.argv) > 2:
        output_path = sys.argv[2] #參數2
        detect_video(YOLO(), video_path, output_path)
    else:
        detect_video(YOLO(), video_path)
