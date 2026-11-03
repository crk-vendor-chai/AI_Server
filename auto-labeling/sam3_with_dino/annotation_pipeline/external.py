import math
import mediapipe as mp
import numpy as np
from .local_cpu import clip,intersection,area,views

class FaceEvidence:
    def __init__(self,path):
        options=mp.tasks.vision.FaceDetectorOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(path),delegate=mp.tasks.BaseOptions.Delegate.CPU),
                     running_mode=mp.tasks.vision.RunningMode.IMAGE,min_detection_confidence=.7,min_suppression_threshold=.3)
        self.detector=mp.tasks.vision.FaceDetector.create_from_options(options)

    def check(self,image,box):
        original=clip(box,image.size);crop=image.crop(original).convert("RGB")
        result=self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(np.asarray(crop))))
        faces=[]
        for detection in result.detections:
            b=detection.bounding_box
            face=[original[0]+b.origin_x,original[1]+b.origin_y,original[0]+b.origin_x+b.width,original[1]+b.origin_y+b.height]
            coverage=intersection(face,original)/max(area(original),1)
            faces.append({"box":face,"coverage":coverage,"score":detection.categories[0].score})
        return {"eligible":any(f["coverage"]>=.1 for f in faces),"faces":faces}

    def close(self):self.detector.close()

class HandEvidence:
    def __init__(self,path):
        base=mp.tasks.BaseOptions(model_asset_path=str(path),delegate=mp.tasks.BaseOptions.Delegate.CPU)
        options=mp.tasks.vision.HandLandmarkerOptions(base_options=base,running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_hands=1,min_hand_detection_confidence=.35,min_hand_presence_confidence=.35,min_tracking_confidence=.5)
        self.detector=mp.tasks.vision.HandLandmarker.create_from_options(options)

    def check(self,image,box):
        tiles=views(box,image.size)
        original=tiles[0]
        for index,tile in enumerate(tiles):
            crop=np.ascontiguousarray(np.asarray(image.crop(tile).convert("RGB")))
            result=self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=crop))
            for landmarks in result.hand_landmarks:
                points=[[tile[0]+p.x*(tile[2]-tile[0]),tile[1]+p.y*(tile[3]-tile[1])] for p in landmarks]
                valid=[p for p in points if all(math.isfinite(v) for v in p) and original[0]<=p[0]<=original[2] and original[1]<=p[1]<=original[3]]
                extent=min(max(p[0] for p in valid)-min(p[0] for p in valid),max(p[1] for p in valid)-min(p[1] for p in valid)) if valid else 0
                if len(valid)>=11 and extent>=3:
                    return {"veto":True,"calls":index+1,"view":index,"landmarks_xy":points,"valid_landmarks":len(valid),"extent_min_pixels":extent,
                            "absence_is_not_GT":True,"handedness_not_used_as_presence_confidence":True}
        return {"veto":False,"calls":len(tiles),"absence_is_not_GT":True}

    def close(self):self.detector.close()
