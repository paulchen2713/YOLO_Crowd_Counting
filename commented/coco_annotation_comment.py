import json
from collections import defaultdict

name_box_id = defaultdict(list)
id_name = dict()
f = open(
    "mscoco2017/annotations/instances_train2017.json",  # 讀取.json檔
    encoding='utf-8')
data = json.load(f)

annotations = data['annotations']
for ant in annotations:
    id = ant['image_id']
    name = 'mscoco2017/train2017/%012d.jpg' % id  # 載入圖片
    cat = ant['category_id']
    # 種類的分類
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11

    name_box_id[name].append([ant['bbox'], cat]) # 輸出boundingbox種類

f = open('train.txt', 'w') # 讀取train.txt檔
for key in name_box_id.keys():
    f.write(key)
    box_infos = name_box_id[key]
    for info in box_infos:
        x_min = int(info[0][0]) # boundingbox左上角x座標
        y_min = int(info[0][1]) # boundingbox左上角y座標
        x_max = x_min + int(info[0][2]) # boundingbox右下角x座標
        y_max = y_min + int(info[0][3]) # boundingbox右下角y座標

        box_info = " %d,%d,%d,%d,%d" % (
            x_min, y_min, x_max, y_max, int(info[1]))#boundingbox座標
        f.write(box_info)#寫入train.txt中
    f.write('\n')
f.close()


