import numpy as np


class YOLO_Kmeans:

    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number
        self.filename = "2012_train.txt"

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1] #所有行的第0個數據*所有行的第1個數據
        box_area = box_area.repeat(k) #重複k次
        box_area = np.reshape(box_area, (n, k)) #將box_area轉成(n,k)的矩陣

        cluster_area = clusters[:, 0] * clusters[:, 1] #所有行的第0個數據*所有行的第1個數據
        cluster_area = np.tile(cluster_area, [1, n]) #向x方向複製n次
        cluster_area = np.reshape(cluster_area, (n, k)) #將cluster_area轉成(n,k)的矩陣

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        #將 boxes所有行的第0個數據重複k次，轉成(n,k)的矩陣
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        #將 clusters所有行的第0個數據向x方向複製n次，轉成(n,k)的矩陣
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix) #兩矩陣取最小的產生新矩陣

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        #將 boxes所有行的第1個數據重複k次，轉成(n,k)的矩陣
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        #將 clusters所有行的第1個數據向x方向複製n次，轉成(n,k)的矩陣
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix) #兩矩陣取最小的產生新矩陣
        inter_area = np.multiply(min_w_matrix, min_h_matrix) #兩矩陣對應位置的元素相乘產生新矩陣

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        #先取陣列最大值，再得出各列的平均值
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k)) #產生未初始化的(box_number, k)矩陣
        last_nearest = np.zeros((box_number,)) #產生零矩陣
        np.random.seed() #產生指定隨機數
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        #隨機抽選不重複的k個clusters
        
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1) #產生最小矩陣(少一行)
            if (last_nearest == current_nearest).all(): #比對兩矩陣的所有對應元素是否相等
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data): #將結果寫入txt檔
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self): #讀取txt檔
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            infos = line.split(" ")
            length = len(infos)
            for i in range(1, length):
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                dataSet.append([width, height])
        result = np.array(dataSet)
        f.close()
        return result

    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number = 9
    filename = "2012_train.txt"
    kmeans = YOLO_Kmeans(cluster_number, filename)
    kmeans.txt2clusters()
