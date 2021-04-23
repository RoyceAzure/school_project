

from dlib import correlation_tracker, rectangle


class CorrelationTracker:
    def __init__(self,bbox,img):
        self.tracker = correlation_tracker()
        self.tracker.start_track(img,rectangle(int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])))
        self.confidence = 0. # measures how confident the tracker is! (a.k.a. correlation score)
  

    def predict(self,img):
        self.confidence = self.tracker.update(img)
        return self.get_state()

    def update(self,bbox,img):
        if bbox != []:
            self.tracker.start_track(img, rectangle(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
            

    def get_state(self):
        pos = self.tracker.get_position()
        return [int(pos.left()), int(pos.top()),int(pos.right()),int(pos.bottom())]