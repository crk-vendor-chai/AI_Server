import math,time
import numpy as np
import torch
from .local_cpu import configure,feature,views
from .kernel import score
from .external import FaceEvidence,HandEvidence

class FallbackVerifier:
    def __init__(self,bundle):
        configure(4);self.assets=bundle
        self.encoder=bundle.load('pooled');self.old=bundle.load('dino_old');self.new=bundle.load('dino_new')
        self.prototype=bundle.document('prototype');self.centers=np.load(bundle.path('centers'))
        self.threshold1=bundle.settings['dino_new']['threshold'];self.threshold3=bundle.settings['prototype']['threshold']
        self.local_threshold=bundle.settings['local_threshold'];self.face_threshold=bundle.settings['face_threshold']
        self.face=None;self.hand=None

    def verify(self,image,source,hands,early_exit=True):
        if len(hands)>1:raise ValueError('At most one hand')
        if source!='dino_fallback' or not hands:return {'hand':hands,'keep':True,'reason':'nonfallback_untouched','encoder_calls':0}
        started=time.perf_counter()
        try:
            image=image.convert('RGB');tiles=views(hands[0],image.size)
            native=min(tiles[0][2]-tiles[0][0],tiles[0][3]-tiles[0][1])
            if native<12:return {'hand':hands,'keep':True,'reason':'tiny_preserved','encoder_calls':0}
            maximum=np.zeros(3,dtype=np.float64);thresholds=[self.face_threshold,self.threshold1,self.threshold3]
            face_eligible=False;count=0
            for index,tile in enumerate(tiles):
                f=feature(self.encoder,image,tile)
                with torch.inference_mode():
                    t=torch.from_numpy(f).unsqueeze(0)
                    old=self.old(t).item();new=self.new(t).item()
                value=[old,new,score(self.centers,self.prototype,f)]
                if not all(math.isfinite(v) for v in value):raise ValueError('Nonfinite score')
                maximum=np.maximum(maximum,value);count+=1
                if index==0 and old<self.face_threshold:
                    if self.face is None:self.face=FaceEvidence(self.assets.path('face'))
                    face_eligible=self.face.check(image,hands[0])['eligible']
                    thresholds[0]=self.face_threshold if face_eligible else self.local_threshold
                if early_exit and all(v>=t for v,t in zip(maximum,thresholds)):break
            rgb_reject=any(v<t for v,t in zip(maximum,thresholds));external=None
            if rgb_reject:
                if self.hand is None:
                    self.hand=HandEvidence(self.assets.path('hand'))
                external=self.hand.check(image,hands[0])
            reject=rgb_reject and not external['veto']
            return {'hand':[] if reject else hands,'keep':not reject,'reason':'ROI_rejected_not_whole_image_absence' if reject else 'hand_or_uncertainty_preserved',
                    'score_lower_bounds':maximum.tolist(),'scores_are_full_max':count==len(tiles),'thresholds':thresholds,
                    'face_eligible':face_eligible,'external':external,'encoder_calls':count,'available_views':len(tiles),
                    'milliseconds':(time.perf_counter()-started)*1000,'whole_image_absence_asserted':False}
        except (RuntimeError,ValueError,OSError) as error:
            return {'hand':hands,'keep':True,'reason':'runtime_error_preserved','error':str(error),'encoder_calls':0}

    def close(self):
        if self.face:self.face.close()
        if self.hand:self.hand.close()
