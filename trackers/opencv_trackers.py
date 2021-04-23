import cv2
class openvc_tracker:
    OBJECT_TRACKERS = {
                "csrt": cv2.TrackerCSRT_create,
                "kcf": cv2.TrackerKCF_create,
                "boosting": cv2.TrackerBoosting_create,
                "mil": cv2.TrackerMIL_create,
                "tld": cv2.TrackerTLD_create,
                "medianflow": cv2.TrackerMedianFlow_create,
                "mosse": cv2.TrackerMOSSE_create,
                }
    def __init__(self, box, frame ,method):
        self.tracker = openvc_tracker.OBJECT_TRACKERS[method]()
        self.tracker.init(frame, box)
    def update(self, frame):
        success, boxes =self.tracker.update(frame)
        return  success, boxes
    def predict(self, frame):
        boxes =self.tracker.update(frame)

        

