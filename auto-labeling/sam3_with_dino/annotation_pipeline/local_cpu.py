"""One CPU inference path shared by feature fitting, calibration and replay."""
import math

import numpy as np
import torch
from PIL import Image


def configure(threads=4):
    torch.set_num_threads(threads)
    torch.backends.mkldnn.enabled = True
    torch.manual_seed(17)


def clip(box, size):
    w,h=size
    return [max(0,math.floor(box[0])),max(0,math.floor(box[1])),min(w,math.ceil(box[2])),min(h,math.ceil(box[3]))]


def area(box):
    return max(0,box[2]-box[0])*max(0,box[3]-box[1])


def intersection(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


def expanded(box, scale, size):
    cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
    w=(box[2]-box[0])*scale;h=(box[3]-box[1])*scale
    return clip([cx-w/2,cy-h/2,cx+w/2,cy+h/2],size)


def views(box, size, local=True):
    original=clip(box,size)
    if min(original[2]-original[0],original[3]-original[1])<1:
        raise ValueError("Empty clipped ROI")
    output=[original]
    if local:
        x,y,x2,y2=original;w=x2-x;h=y2-y
        for oy in [0,.25,.5]:
            for ox in [0,.25,.5]:
                tile=clip([x+ox*w,y+oy*h,x+(ox+.5)*w,y+(oy+.5)*h],size)
                if min(tile[2]-tile[0],tile[3]-tile[1])>=12 and tile not in output:
                    output.append(tile)
    return output


def tensor(image, box):
    b=clip(box,image.size)
    crop=image.crop(b)
    if min(crop.size)<1:raise ValueError("Empty ROI")
    scale=224/max(crop.size)
    crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),Image.Resampling.BILINEAR)
    out=Image.new("RGB",(224,224),(124,116,104))
    out.paste(crop,((224-crop.width)//2,(224-crop.height)//2))
    value=torch.from_numpy(np.asarray(out,dtype=np.float32)/255.).permute(2,0,1)
    return (value-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]


def feature(encoder, image, box):
    # Always batch1, including during fitting and replay; no CPU/GPU path split.
    with torch.inference_mode():
        return torch.nn.functional.normalize(encoder(tensor(image,box).unsqueeze(0)),p=2,dim=1).numpy()[0]


def gate(source, hands, score, threshold, native_min_side, veto=False):
    if len(hands)>1:raise ValueError("At mostone hand")
    if source!="dino_fallback" or not hands:
        return hands,{"keep":True,"review_pending":False,"reason":"original_nonfallback_untouched"}
    if native_min_side<12:
        return hands,{"keep":True,"review_pending":True,"reason":"tiny_preserved"}
    if not math.isfinite(score):
        return hands,{"keep":True,"review_pending":True,"reason":"invalid_score_preserved"}
    if veto:
        return hands,{"keep":True,"review_pending":score<.8,"reason":"external_positive_hand_veto"}
    if score<threshold:
        return [],{"keep":False,"review_pending":False,"reason":"ROI_rejected_not_whole_image_absence"}
    return hands,{"keep":True,"review_pending":score<.8,"reason":"uncertain_preserved" if score<.8 else "hand_supported"}
