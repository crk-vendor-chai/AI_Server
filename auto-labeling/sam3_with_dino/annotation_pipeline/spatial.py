import math
import numpy as np
import torch
from PIL import Image

def clip(b,size=(640,480)):
    return [max(0,math.floor(b[0])),max(0,math.floor(b[1])),min(size[0],math.ceil(b[2])),min(size[1],math.ceil(b[3]))]

def context(b,scale=1.35):
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2;w,h=max(96,(b[2]-b[0])*scale),max(96,(b[3]-b[1])*scale)
    return clip([cx-w/2,cy-h/2,cx+w/2,cy+h/2])

def tensor(image,box):
    b=context(box);crop=image.crop(b)
    if min(crop.size)<1:raise ValueError('Empty context')
    w,h=crop.size;s=224/max(w,h);rw,rh=max(1,round(w*s)),max(1,round(h*s));px,py=(224-rw)//2,(224-rh)//2
    patch=Image.new('RGB',(224,224),(124,116,104));patch.paste(crop.resize((rw,rh),Image.Resampling.BILINEAR),(px,py))
    x=torch.from_numpy(np.asarray(patch,dtype=np.float32)/255).permute(2,0,1)
    return (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]

def grid_mask(box,ctx,size=14):
    w,h=ctx[2]-ctx[0],ctx[3]-ctx[1];scale=224/max(w,h);rw,rh=max(1,round(w*scale)),max(1,round(h*scale));px,py=(224-rw)//2,(224-rh)//2
    b=[(box[0]-ctx[0])*rw/w+px,(box[1]-ctx[1])*rh/h+py,(box[2]-ctx[0])*rw/w+px,(box[3]-ctx[1])*rh/h+py]
    lo=np.arange(size)*224/size;hi=(np.arange(size)+1)*224/size
    return torch.from_numpy(((hi[:,None]>b[1])&(lo[:,None]<b[3])&(hi[None,:]>b[0])&(lo[None,:]<b[2])).astype(np.float32))

def presence_logit(logits,support):
    value=logits.flatten(1).masked_fill(support.flatten(1)<=0,-30)
    # Small ROI may have fewer than3 supported cells: maximum remains an explicit term.
    top=torch.topk(value,k=3,dim=1).values;valid=top>-29
    mean=(top*valid).sum(1)/valid.sum(1).clamp_min(1)
    return .5*value.max(1).values+.5*mean
