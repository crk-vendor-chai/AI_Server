import math
from . import spatial as H

def patch_boxes(box):
    b=H.clip(box)
    w,h=b[2]-b[0],b[3]-b[1]
    if min(w,h)<=0: return []
    if max(w,h)<=160: return [b]
    side=max(160,int(math.ceil(math.sqrt(w*h/6))))
    def starts(lo,hi):
        if hi-lo<=side: return [lo]
        seq=list(range(lo,hi-side+1,max(1,int(side*.625))))
        return list(dict.fromkeys([*seq,hi-side]))
    tiles=[[x,y,min(x+side,b[2]),min(y+side,b[3])] for y in starts(b[1],b[3]) for x in starts(b[0],b[2])]
    if len(tiles)>12:
        tiles=[tiles[i] for i in sorted({round(j*(len(tiles)-1)/11) for j in range(12)})]
    return [b,*[t for t in tiles if t!=b]]
