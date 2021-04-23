import numpy as np
class Filter:
    def __init__(self, w, h, max_box_to_draw = 10 , min_score_thresh = 50, filter_number = 10):
        self.max_box_to_draw = max_box_to_draw
        self.min_score_thresh = min_score_thresh
        self.filter_number = filter_number
        self.H, self.W = (h,w)
    def Filter_people_box(self, boxes, classes, scores):#過濾tensor偵測出來的物件
        total_count = 0
        rect = []
        iterator =0
        '''
        初始化  total_count計算總偵測物件數
        rect 用來裝過濾完的物件  待下一個步驟使用
        iterator   用來計算到底過濾了多少物件  數字要小於max_box_to_draw  因為只要過濾15個
        '''
        filt_array=np.squeeze(np.array(np.where(classes==1)), axis = 0)[:self.filter_number]
        size = len(filt_array)

        filt_scores = scores[filt_array]
        print("filt_scores:{}".format(filt_scores))
        '''
        np.where(classes==1) => 把classes裡面是1的(人)選出來 會回傳一個list, value == index
        np.array=>把回傳的結果轉成nparray
        np.squeeze(axis = 0)=>把nparray 0位置的為度去掉 為了下個步驟用
        [:filter_number]=> 取前15個
        filt_array =>現在filt_array裡面裝了15個偵測到人的物件的index  (已經有照分數高到低排序)
        filt_scores = scores[filt_array]=> 用filt_array的index 把對映的分數從scores取出來

        現在filt_scores, filt_array 同一個index上放的是對應的 (分數 , 物件index)
        '''

        for i in range(size):                                       #size就是filt_array的長度
            if  filt_scores[i]*100 > self.min_score_thresh:              #如果分數有大於門檻
                iterator = iterator+1                               
                total_count+=1                                      #表示有偵測到人 所以+1
                box = boxes[filt_array[i]] * np.array([ self.H, self.W, self.H, self.W,])
                (startY, startX, endY, endX) = box.astype("int") 
                #print("lefttype:{}".format(type(left)))
                #print("left, right, top, bottom:{}".format((left, right, top, bottom)))
                rect.append((startX, startY, endX, endY))   #把換算完的4個座標放入rect
                if iterator>=self.max_box_to_draw:  #若已經處理了15個物件  就該停止了
                    break
        return rect,total_count  #把所有處理完的 物件的4個座標傳回去

    def predict_Filter_people_box(self, boxes, classes, scores):#過濾tensor偵測出來的物件

        total_count = 0
        rects = []
        iterator =0
        adds = list()
        filt_array=np.squeeze(np.array(np.where(classes==1)), axis = 0)[:self.filter_number]
        size = len(filt_array)
        filt_scores = scores[filt_array]

        for i in range(size):                                       #size就是filt_array的長度
            if  filt_scores[i]*100 > self.min_score_thresh:              #如果分數有大於門檻
                iterator = iterator+1                               #
                total_count+=1                                      #表示有偵測到人 所以+1
                box = boxes[filt_array[i]] * np.array([ self.H, self.W, self.H, self.W,])
                (startY, startX, endY, endX) = box.astype("int")

                adds.append((startX, startY, endX-startX,endY-startY))
                rects.append((startX, startY, endX, endY))   #把換算完的4個座標放入rect
                if iterator>=self.max_box_to_draw:  #若已經處理了15個物件  就該停止了
                    break
        return rects,total_count,adds   #把所有處理完的 物件的4個座標傳回去