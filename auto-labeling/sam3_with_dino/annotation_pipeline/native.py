import numpy as np
import torch
from PIL import Image
from . import spatial as H

def tensor(im,box):
    crop=im.crop(H.clip(box));w,h=crop.size
    if min(w,h)<1:raise ValueError('Empty proposal')
    scale=224/max(w,h);rw,rh=max(1,round(w*scale)),max(1,round(h*scale))
    out=Image.new('RGB',(224,224),(124,116,104));out.paste(crop.resize((rw,rh),Image.Resampling.BILINEAR),((224-rw)//2,(224-rh)//2))
    x=torch.from_numpy(np.asarray(out,dtype=np.float32)/255).permute(2,0,1)
    return (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]

def matches(feature,bank,thresholds):
    similarity=np.asarray(feature,dtype=np.float32)@bank.T
    return bool(np.any(similarity>=thresholds+np.float32(1e-5)))
