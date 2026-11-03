import numpy as np
import torch
from PIL import Image,ImageEnhance
from . import spatial as H

def tensor(im,box,brightness=1.):
    b=H.clip(box)
    if min(b[2]-b[0],b[3]-b[1])<=0:raise ValueError('Empty proposal')
    crop=im.crop(b);w,h=crop.size;scale=96/max(w,h);rw,rh=max(1,round(w*scale)),max(1,round(h*scale))
    out=Image.new('RGB',(96,96),(124,116,104))
    crop=crop.resize((rw,rh),Image.Resampling.BILINEAR)
    if brightness!=1.:crop=ImageEnhance.Brightness(crop).enhance(brightness)
    out.paste(crop,((96-rw)//2,(96-rh)//2))
    return torch.from_numpy(np.asarray(out,dtype=np.float32)/255).permute(2,0,1)
