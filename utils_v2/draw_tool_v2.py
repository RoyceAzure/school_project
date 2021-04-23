import numpy as np
import cv2
class draw_tool:
    def __init__(self):
        pass
    def draw_filed(self, frame, filed_info):
        if len(filed_info) !=0:                                      #如果dict裡面有東西 才要畫
            for filed in filed_info:            #依序取出每個邊框資訊  並用draw畫上
                ID = "ID:{}".format(str(filed.filed_id))
                total_count = "total_count:{}".format(str(filed.total_count))
                current_count ='current_count:{}'.format(str(filed.current_count))
                total_stay_time ='total_staytime:{}'.format(str(filed.total_stay_time)) 
                avg_stay_time ='avg_stay_time:{}'.format(str(filed.avg_stay_time)) 
                # self.draw.text((ix+1,iy+1),text = show_text,font = self.font,fill = filed_info[(ix,iy,x,y)]["bordercolor"])
                texts = [ ID, total_count, current_count, total_stay_time, avg_stay_time]
                i = 1
                for text in texts:
                    cv2.putText(frame, text, (filed.left +1, filed.top+i*15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    i+=1
                cv2.rectangle(frame, (filed.left, filed.top), (filed.right, filed.bottom), (0, 0, 0), 1)