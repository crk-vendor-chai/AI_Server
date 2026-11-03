import hashlib,json,math

def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])

def inter(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))

def iou(a,b):
    x=inter(a,b);return x/max(1e-9,area(a)+area(b)-x)

def clip(b):return [max(0.,min(640.,b[0])),max(0.,min(480.,b[1])),max(0.,min(640.,b[2])),max(0.,min(480.,b[3]))]

def expand(b,scale=1.5,minimum=64):
    x,y=(b[0]+b[2])/2,(b[1]+b[3])/2;w,h=max(minimum,(b[2]-b[0])*scale),max(minimum,(b[3]-b[1])*scale)
    return clip([x-w/2,y-h/2,x+w/2,y+h/2])

def sample(rows,n):
    rows=sorted(rows,key=lambda r:r['id'] if 'id' in r else r['key'])
    return rows if len(rows)<=n else [rows[round(j*(len(rows)-1)/(n-1))] for j in range(n)]

def token(key,box):return hashlib.sha256(json.dumps([key,box],separators=(',',':')).encode()).hexdigest()[:24]
