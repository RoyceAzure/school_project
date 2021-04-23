import numpy as np
from collections import defaultdict
import cv2
import os
import pandas as pd
class S_Filed_manager:
    def __init__(self, Id, cordinate, color):
        self.filed_id = Id
        self.dected_people = defaultdict(dict) #記錄所有來過此區的人 [ID] = { 'stay_time', 'is_still_in' }
        self.total_count = 0                   #所有來過此區人的數量
        self.current_count = 0                 #當前區域內有多少人
        self.color = color                     #此區域的顏色
        self.left,self.top,self.right,self.bottom = cordinate   #記錄此區的座標
        self.total_stay_time = 0                                #記錄所有人在此區的總停留時間
        self.avg_stay_time = 0
        self.hot_sopt_weight = 0
        self.total_delete = 0             
    def is_infiled(self, centroid):                             #檢查傳入的人的質心位置是不是在本區域內
        return centroid[0] < self.right and centroid[0] > self.left and centroid[1]> self.top and centroid[1] < self.bottom
    def is_self_dected(self):                                   #本區是否有人來過 (是否有記錄任何資料)
        if self.dected_people:
            return True
        else:
            return False
    def deregister(self, Id):
        self.total_stay_time+=int(self.dected_people[Id]['stay_time'])
        self.dected_people[Id]['is_still_in'] = 0
        self.total_delete+=1

    def save_total_count(self, dir_path):
        total_count = self.current_count + self.total_delete

        title = 'filed_total_count.csv'
        save_data = [[ self.filed_id, total_count]]
        select =  pd.DataFrame(save_data, columns = ["fid","total_count"])
        path = dir_path + title
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # print("filed_save_path_tootal count:{}".format(path))

        if os.path.isfile(path):
            with open(path, 'a') as f:
                select.to_csv(f, header=False, index=0)         
        else:
            with open(path, 'a') as f:    
                select.to_csv(f, index=0)



        self.total_delete = 0

    def save_final_save(self , final_save):
        save_data = [[self.filed_id, self.total_count, self.avg_stay_time, self.hot_sopt_weight]]
        select =  pd.DataFrame(save_data, columns = ["fid","total_count",  "avg_time", "hot_spot"])
        if os.path.isfile(final_save):
            with open(final_save, 'a') as f:
                select.to_csv(f, header=False, index=0)         
        else:
            with open(final_save, 'a') as f:    
                select.to_csv(f, index=0)


    def count_in_filed(self, trackers):                        #用有配對到的來計算區域內人數 selected_cen==成功配對的list
        count = 0
        Ids = [ trk.track_id for trk in trackers]                                  #取出所有配對到的物件ID
        object_keys = self.dected_people.keys()                    #取出本身記錄的所有物件ID
        for Id in Ids:
            if Id in object_keys:                                  #如果配對到的ID 是本身有記錄的
                if self.dected_people[Id]['is_still_in'] == 1:     #進一步在確認是否還在區域內
                    count+=1                                       #是則count +1
        return count

    def update(self, trackers, frame, time_proid, base_save, fps = 30):  
        if trackers:
            id_and_cen = [[trk.track_id, trk.centroid] for trk in trackers]
            ids = [ trk.track_id for trk in trackers]                                        #如果selected_cen有東西(有成功配對) 才會做進階判斷
            for ID, centroid in id_and_cen:              # 取出配對到的每組ID , 質心位置
                if self.is_infiled(centroid):                      #檢查質心位置是否在區域內
                    if ID not in self.dected_people.keys():  #若質心位置在區域內  但是沒有該物件的註冊資訊, 表示他是新來的  要幫他創建dict記錄資料
                        self.dected_people[ID] = { 'stay_frame': 1, 'is_still_in' : 1, 'is_counted' : False, 'stay_time' : 0}  # 用他的id創造一個dict
                        self.current_count+=1                                          #目前區域的current_count+1
                        self.total_count+=1                                    #目前區域的total_count+1 因為是新來的  不必考慮重複 所以總數加1  
                    else:                                                     #表示該ID已經有註冊(表示已經來過)
                        self.dected_people[ID]['is_still_in'] = 1             #改變該物件狀態為"在區域內" 
                        self.dected_people[ID]['stay_frame']+=1
                        self.dected_people[ID]['stay_time'] = float(round(self.dected_people[ID]['stay_frame']/fps,3))
            if self.is_self_dected():
                all_ids = list(self.dected_people.keys())   
                # print("in field{} all ids:{} ".format(self.filed_id, all_ids))
                # print("all teacker ids:{} ".format(ids))
                # print("len of ids:{}".format(ids))                                                      #把本身所有記錄的物件ID取出來  用來比對質心座標看是否離開
                for Id in all_ids:
                    for tid, cen in id_and_cen:             #一個一個檢查
                        if Id == tid:                                  #如果ID有被配對到  才需要檢查, 沒配對到的不理他
                            if not self.is_infiled(cen):            #檢查是否有在區域內  
                                if self.dected_people[Id]['is_still_in'] == 1: #若檢查完座標不在區域內  且該物件原本是在區域內  則代表意思為離開
                                    self.deregister(Id)                        #離開物的物件呼叫deregister  改變狀態
                    if Id not in ids:
                        self.deregister(Id)                           # objects是記錄所有偵測到的物件資訊  若裡面沒有該ID 表是該物件已經被許銷註冊(走出螢幕)所以要刪除
                        self.save_csv_filed(Id, time_proid, base_save)
                        del self.dected_people[Id]
                        # print(" delete id:{}".format(Id))
                self.avg_stay_time = round(self.total_stay_time/self.total_count,3) #計算平均停留時間
            self.current_count = self.count_in_filed(trackers)
            save_list = self.check_hot_spot()
            if save_list:
                id_and_box = [[trk.track_id, trk.bbox] for trk in trackers]
                for Ids in  save_list:
                    for Idt, box in id_and_box:
                        if Ids == Idt:
                            self.save_peopel_img(Ids, frame,box, time_proid, base_save)
        else:
            if self.is_self_dected():
                self.dected_people.clear()
    def check_hot_spot(self, stay_threhold = 10):
        save_image_list = list()
        for Id, value in self.dected_people.items():
            if value['stay_time'] > stay_threhold and not value['is_counted']:
                self.hot_sopt_weight+=1
                value['is_counted'] = True
                save_image_list.append(Id)
        return save_image_list

    def save_peopel_img(self, Id, frame, box,time_proid, base_save):
        base = base_save + time_proid 
        title = 'fid_{}_pid_{}.jpg'.format(self.filed_id, Id)
        img = frame[box[1]:box[3],box[0]:box[2]]
        path = base + "/" + title
        # print("in save_peopel_img!!")
        # print(path)
        os.makedirs(base, exist_ok=True)
        cv2.imwrite(path, img)



    def testdata(self):
        for Id, vallue in self.dected_people.items():
            print("filed id:{} dected_people dict id: {}, value:{}".format(self.filed_id, Id, vallue))

    def save_csv_filed(self, Id, time_proid, base_save):
        save_list = [[self.filed_id,Id, self.dected_people[Id]['stay_time']]]
        select = pd.DataFrame(save_list, columns = ["fid","pid","stay_time"])
        title = 'filed_of_missing_peopel.csv'
        dir_path = base_save + time_proid
        path = dir_path + '/' + title
        if os.path.isfile(path):
            with open(path, 'a') as f:
                select.to_csv(f, header=False, index=0)         
        else:
            with open(path, 'a') as f:    
                select.to_csv(f, index=0)






class All_filed:
    STANDARD_COLORS = [                                                   #每個區域的顏色
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
    ]
    def __init__(self, video_name, time_period):
        self._ix,self._iy,self._x,self._y = (-1,-1,-1,-1)                         #記錄滑鼠做標 初始化為-1
        self._drawing = False
        self._filed_info = list()                                      #記錄所有區域的資訊 key=id, value=區域作標
        self._fileNextId = 1
        self.time_proid = '0_{}'.format(time_period)
        self.base_save = 'save/time_split/{}/'.format(video_name)
        self.final_save = 'save/final_info_{}.csv'.format(video_name)
    def updateI(self, cor):                                                #當滑鼠左鍵按下去  記錄座標存在 self._ix , self._iy
        self._drawing = False
        self._ix,self._iy = cor
    def updateFiled(self, cor):                                            #當滑鼠左鍵放開  記錄座標存在 self._x , self._y
        self.drawing = True
        self._x,self._y = cor                                              #線在有區域的4個座標  可以創造一個區域了
        color = All_filed.STANDARD_COLORS[self._fileNextId % len(All_filed.STANDARD_COLORS)]
        self._filed_info.append(S_Filed_manager(self._fileNextId,
                                                (self._ix, self._iy, self._x, self._y),
                                                self._fileNextId % len(All_filed.STANDARD_COLORS)))
        self._fileNextId+=1  

    def save_all_count(self):
        path = self.base_save + self.time_proid + '/'
        for filed in self._filed_info:
            filed.save_total_count(path)

    def final_save_all(self):
        for filed in self._filed_info:
            filed.save_final_save(self.final_save)


    def get_all_filed(self):                                               #回傳有所有區域資訊的dict
        return self._filed_info
    
    def get_draw(self):
        return self._drawing                                               #畫區域邊框用的
    

    def testdata(self):
        print("in fm ix:{},iy:{},x:{},y:{}".format(self._ix,self._iy,self._x,self._y))
        
    def dected_per_frame(self, trackers, frame):          #fm程式進入點
        if self._filed_info:                                    #如果有記錄任何區域  則每個區域取出來做update   
            for filed in self._filed_info:
                filed.update(trackers, frame, self.time_proid, self.base_save)
                filed.testdata()
                
    def is_has_filed(self):                                    #回傳self._filed_info 是否有東西
        if self._filed_info:
            return True
        else:
            return False
        
         