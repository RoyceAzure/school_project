

# Import python libraries
import numpy as np
from trackers import kalman_filter, kalman_tracker, correlation_tracker, opencv_trackers
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment
from sklearn.utils.linear_assignment_ import linear_assignment
import os
import pandas as pd

class Track:  #用來記錄單一tracker的資料


    CV_OBJECT_TRACKERS = [
                "csrt","kcf","boosting","mil","tld","medianflow","mosse"]
    def __init__(self, bbox, trackIdCount, method , frame = None):
        self.track_id = trackIdCount   #id 
        self.bbox = bbox   #邊框
        self.centroid = self.cal_centroid(bbox)  #當下質心
        self.foot_print = self.cal_foot()
        self.skipped_frames = 0   #多少frame沒偵測到
        self.path_centroids = []  #記錄軌跡
        self.method = method      #記錄使用的追蹤方法
        self.confidence = 15.0     
        self.counted = False      #是否有被計算過 (出入口數人要用)
        self.path_centroids.append(self.foot_print)  #把第一個質心加入到路徑

        if self.method in Track.CV_OBJECT_TRACKERS:  #根據不同分法建立tracker
            self.track_ob = opencv_trackers.openvc_tracker(frame, bbox, method)
        if self.method == "dlib":
            self.track_ob = correlation_tracker.CorrelationTracker(bbox, frame)
        if self.method == "kalman":
            self.track_ob = kalman_filter.KalmanFilter()


    def cal_centroid(self, box):
            cX = int((box[0] + box[2]) // 2.0)
            cY = int((box[1] + box[3]) // 2.0)
            return [cX, cY]
    def cal_foot(self):
        cx,cy = self.centroid
        fx = cx
        fy = self.bbox[3]
        return [fx, fy]
    def save_str(self):   #用字串方是儲存資料
        return "{},{},{},{}\r\n".format(self.track_id, self.bbox, self.centroid, self.path_centroids)

    def save_csv(self, time_proid, path):
        # print("in save_csv!!!!!")
        dir_path = path + time_proid
        title = "/missing_pid{}.csv".format(self.track_id)
        # print("dir_path:{}".format(dir_path))
        
        save_path = path + time_proid + title
        select = pd.DataFrame(self.path_centroids, columns = ["x", "y"])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # print("save_path:{}".format(save_path))
        select.to_csv(save_path , index=0 ) 
         





class Trackers: #用來記錄所有tracker的資料

    trackIdCount = 0
    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length, leave_limit ,frame_info, video_name, time_proid) :

        self.dist_thresh = dist_thresh    #最大配對距離
        self.max_frames_to_skip = max_frames_to_skip  
        self.max_trace_length = max_trace_length  #最多記錄多少路徑
        self.leave_limit = leave_limit     #離開的邊界
        self.trackers = []                 #所有的tracker
        self.time_proid ='0_{}'.format(time_proid)
        self.W,self.H = frame_info
        self.save_path = 'save/time_split/{}/'.format(video_name)
        self.final_save_path = 'save/final_peopel_save_{}.csv'.format(video_name)
        self.total_delete = 0
        self.count_init()
    def create_track(self, detection, frame, method):  #detection [left top right bottom] 
            track = Track(detection, Trackers.trackIdCount, method, frame)
            Trackers.trackIdCount += 1
            self.trackers.append(track)


    def count_init(self):
        path = "save/total_count.txt"
        if os.path.isfile(path):
            with open(path , "r") as f:
                data = f.read()
                self.totalLeft = int(data.split(",")[0])
                self.totalRight = int(data.split(",")[1])
        else:
            self.totalLeft, self.totalRight = (0,0)


    def  y_deraction(self):
        for trk in self.trackers:
            x = [c[0] for c in trk.path_centroids]
            centroid = trk.centroid
            direction = int(centroid[0] - np.mean(x)) #計算方向
            if not trk.counted:
                if direction  < 0 and self.W//2 and centroid[0] < self.W//2:
                    self.totalLeft += 1 
                    trk.counted = True

                elif direction > 0 and self.W//2 and centroid[0] > self.W//2:
                    self.totalRight += 1
                    trk.counted = True

    def save_left_right_count(self):
        path = "save/total_count.txt"
        with open(path, "w") as f:
            data= "{},{}".format(self.totalLeft, self.totalRight)
            f.write(data)





    def Organize_track_update(self, track, updatebbox): #更新track
            # print("in Organize_track_update ")
            track.bbox = updatebbox                                     
            track.centroid = track.cal_centroid(track.bbox)
            track.foot_print = track.cal_foot()
            track.path_centroids.append(track.foot_print)
            if len(track.path_centroids) > self.max_trace_length :
                track.path_centroids.pop(0)

    def save_total_people(self):
        # print(" i n save total _people!!!!")
        title = "total_people_count.csv"
        dir_path =  self.save_path + self.time_proid + '/'
        path = dir_path + title
        total_number = [len(self.trackers) + self.total_delete]
        # print("total_number:{}".format(total_number))

        select = pd.DataFrame(total_number, columns = ["total_people_count"])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # print("select:{}".format(select))
        # print("path:{}".format(path))
        select.to_csv(path, index = 0)
        self.total_delete = 0

    # def final_save_all(self):
    #     for track in self.trackers:
    #         track.final_save(self.final_save_path)

    def checked_leaved(self):  #看是否離開
        if self.trackers:
            leaves = []
            for i, track in enumerate(self.trackers):
                x = [c[0] for c in track.path_centroids]
                directionx = track.centroid[0] - np.mean(x)
                y = [c[1] for c in track.path_centroids]
                directiony = track.foot_print[1] - np.mean(y)
                if (directionx < 0 and track.centroid[0] < self.leave_limit) or (directionx > 0 and track.centroid[0] > (self.W - self.leave_limit)) or (directiony < 0 and track.centroid[1] < self.leave_limit) or (directiony > 0 and track.centroid[1] > (self.H-self.leave_limit)):
                    leaves.append(i)

            amend = 0
            print(" in checked leave")
            if leaves:  
                for i in leaves:
                    # print("idex about to pop:{}".format(i))
                    ID = self.trackers[(i-amend)].track_id
                    cen = self.trackers[(i-amend)].centroid
                    print("id:{} about to be delete !!".format(ID))
                    # print(" Leave Object about to be delete id:{}, cen:{}".format(ID, cen))
                    self.trackers[i-amend].save_csv(self.time_proid, self.save_path)
                    self.trackers.pop(i-amend)
                    self.total_delete+=1
                    amend+=1

    def check_skip_frame(self):  #是否有超過skip_frame
        # print("in check skip frame")
        out_of_range = []
        for i, track in enumerate(self.trackers):
            # print("ID:{}, skipframe:{}".format(track.track_id,track.skipped_frames))
            if track.skipped_frames >= 3:
                out_of_range.append(i)
            elif track.skipped_frames >= self.max_frames_to_skip and track.confidence < 10:
                out_of_range.append(i)
        # print(" out_of_range:{}".format(out_of_range)) 
        amend = 0     
        for i in out_of_range:
            # print("idex about to pop:{}".format(i))
            ID = self.trackers[(i-amend)].track_id
            cen = self.trackers[(i-amend)].centroid
            # print(" out_skip_frame Object about to be delete id:{}, cen:{}".format(ID, cen))
            self.trackers[i-amend].save_csv(self.time_proid, self.save_path)  
            self.trackers.pop(i-amend)
            self.total_delete+=1
            amend+=1      
        


                                        #----------------以上為找出沒有註冊的dection 然後全部註冊-----------------------------#
 
    def dis_match_2(self, detections):   #用這個  老樣子
        inputCentroids = np.zeros((len(detections), 2), dtype="int")
        trackCentroids = np.zeros((len(self.trackers), 2), dtype="int")
        trackbox = np.zeros((len(self.trackers), 4), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(detections):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        for i,track in enumerate(self.trackers):
            trackCentroids[i] = track.centroid
            trackbox[i] = track.bbox
        # print("trackbox:{}".format(trackbox))
        # print("trackCentroids:{}".format(trackCentroids))
        # print("detections:{}".format(detections))
        # print("inputCentroids:{}".format(inputCentroids))
        D = dist.cdist(trackCentroids, inputCentroids)
        # print("D:{}".format(D))
        rows = D.min(axis=1).argsort()
        # print("rows:{}".format(rows))
        cols = D.argmin(axis=1)[rows]
        # print("cols:{}".format(cols))

        usedRows = set()
        usedCols = set()
        assignment = []
        
        for i in range(len(self.trackers)):
            # print("i:{}".format(i))
            assignment.append(-1)
        for i, (row, col) in enumerate(zip(rows, cols)):
            if row in usedRows or col in usedCols or D[row, col] > self.dist_thresh:
                continue
            assignment[rows[i]] = cols[i]
            usedRows.add(row) 
            usedCols.add(col)
        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)
        # print("unusedRows:{}".format(unusedRows))
        # print("unusedCols:{}".format(unusedCols))
        for ID in unusedRows:
            self.trackers[ID].skipped_frames+=1
        return assignment

    def match_3(self, detections, iou_threshold = 0):
        inputCentroids = np.zeros((len(detections), 2), dtype="int")
        trackCentroids = np.zeros((len(self.trackers), 2), dtype="int")      
        trackbox = np.zeros((len(self.trackers), 4), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(detections):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY) 
        for i,track in enumerate(self.trackers):
            trackCentroids[i] = track.centroid
            trackbox[i] = track.bbox
        D = dist.cdist(trackCentroids, inputCentroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        usedRows = set()
        usedCols = set()
        assignment = []
        for i in range(len(self.trackers)):
            # print("i:{}".format(i))
            assignment.append(-1)
        for i, (row, col) in enumerate(zip(rows, cols)):
            if row in usedRows or col in usedCols or D[row, col] > self.dist_thresh:
                continue
            assignment[rows[i]] = cols[i]
            usedRows.add(row) 
            usedCols.add(col)
        unusedRows = set(range(0, D.shape[0])).difference(usedRows)  #row=>tracker   col => dections
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)
        print("detections:{}".format(detections))
        print("unusedCols:{}".format(unusedCols))
        unuseddections = [ detections[i] for i in  unusedCols]   #unuseddections放的是 bbox
        check_table = [ i for i in  unusedCols]  
        print("unuseddefctions:{}".format(unuseddections))
        iou_matrix = np.zeros((len(unuseddections),len(self.trackers)),dtype=np.float32)
        for o,obj in enumerate(unuseddections):
            for t,trk in enumerate(self.trackers):
                iou_matrix[o,t] = self.IOU( obj, trk.bbox) #計算每個trk與obj的IOU並且將值存在iou_matrix裡面(依編號存)
        matched_indices = linear_assignment(-iou_matrix)
        for m in matched_indices:                      #m [unmetch_dect, trk]
            if(iou_matrix[m[0],m[1]] > iou_threshold):
                unusedCols.remove(check_table[m[0]]) 
        print("unusedCols after iou:{}".format(unusedCols))
        for ID in unusedRows:
            self.trackers[ID].skipped_frames+=1
        return assignment, list(unusedCols)

    def IOU(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
    
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou


    def data_association(self, detections, frame, iou_threshold = 0): #人愈小 threshold值要愈小，總之threshold值愈小就辨識新物體更難，threshold值愈大辨識到新物體更容易
        iou_matrix = np.zeros((len(detections),len(self.trackers)),dtype=np.float32)
        # self.predict_update(frame)
        for o,obj in enumerate(detections):
            for t,trk in enumerate(self.trackers):
                iou_matrix[o,t] = self.IOU( obj, trk.bbox) #計算每個trk與obj的IOU並且將值存在iou_matrix裡面(依編號存)
        # print("iou_matrix")
        # print(iou_matrix)
        matched_indices = linear_assignment(-iou_matrix)
        # print("matched_indices")
        # print(matched_indices)
        usedDections = set()
        usedTrackers = set()


        matches = []
        for m in matched_indices:                      #m [dect, trk]
            if(iou_matrix[m[0],m[1]] > iou_threshold):
                matches.append(m.reshape(1,2))
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0) #令matches作為一個個row排序這樣
        assignment = list()
        for i in range(len(self.trackers)):
            # print("i:{}".format(i))
            assignment.append(-1)
        for match in matches:
            if match[1] in usedTrackers or match[0] in usedDections:
                continue
            assignment[match[1]] = match[0]
            usedDections.add(match[0])
            usedTrackers.add(match[1])
            unusedTrackers = set(range(0, len(self.trackers))).difference(usedTrackers)
            for ID in unusedTrackers:
                self.trackers[ID].skipped_frames+=1
        return assignment

    def update_kalman(self, assinment):  #沒有用到
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                                            detections[assignment[i]], 1)
            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                                            np.array([[0], [0]]), 0)

            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) -
                               self.max_trace_length):
                    del self.tracks[i].trace[j]

            self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].KF.lastResult = self.tracks[i].prediction

    def debug(self):
        for track in self.trackers:
            print("track information  id:{} ,{} ,{}, skip frame:{}, confidencd:{}".format(track.track_id, 
            track.bbox, track.centroid, track.skipped_frames, track.confidence))

    def detect_Update(self, detections, frame, method ): #偵測時要用的update
        track_len = len(self.trackers)
        if track_len == 0:  #若沒有追蹤物件則全部新增 
            for det in detections:
                self.create_track(det, frame, method)
        else :
            if detections:      #有偵測到才會跑
                # print("AFTER MATCHING!!!!")
                
                assignment, unusedCols = self.match_3(detections)  #assignment是配對完結果 [1 3 2] 表示 (0,1) (1,3) (2,2)
                # assignment = self.data_association(detections, frame)
                for i in range(len(assignment)):                      #更新追蹤物件
                    if assignment[i] != -1:
                        if self.trackers[i].confidence < 10:
                            self.Organize_track_update( self.trackers[i], detections[assignment[i]])
                        self.trackers[i].skipped_frames =0
                        self.trackers[i].track_ob.update(self.trackers[i].bbox,frame)
                # un_assigned_detects = []
                # for i in range(len(detections)):
                #         if i not in assignment:
                #             un_assigned_detects.append(i)
                # print("un_assigned_detects:{}".format(un_assigned_detects))
                dect_len = len(unusedCols)
                if dect_len !=0 :
                    for i in range(len(unusedCols)):
                        self.create_track(detections[unusedCols[i]], frame, method)
                # print("AFTER MATCHING!!!!")
                
            else:
                # print("all skipframe +=1")
                for track in self.trackers:
                    track.skipped_frames+=1
            self.debug()   
            self.check_skip_frame()
            # print("all trackers ids!!")
            # for track in self.trackers:
            #     print(str(track.track_id) + ',', end='')

            
                                            #----------------以上為配對完後  更新tracker centroid-----------------------------#
        
    def get_trackers(self):
        return self.trackers
    
    def predict_update(self, frame):  #追蹤時用的update
        for track in self.trackers:
            bbox = track.track_ob.predict(frame)
            track.confidence = track.track_ob.confidence
            self.Organize_track_update(track, bbox)
            # print("in predict_update box :{}".format(track.bbox))
        # self.debug()
