"""Fresh RGB ANNOTATION preparation: original gates, guarded restoration, final non-target gate."""
import argparse,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .bundle import Bundle
from .gates import Features,Gates
from .geometry import *
from .temporal import choose_restore,choose_hand,neighbours,one_direction


def read_lines(path):
    with Path(path).open() as f:return [json.loads(s) for s in f if s.strip()]
def write_lines(path,rows):
    with Path(path).open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def boxes(text,product_id):
    result={'hand':[],'product':[]}
    for line in text.splitlines():
        if not line.strip():continue
        label,cx,cy,w,h=line.split();label=int(label);cx,cy,w,h=map(float,[cx,cy,w,h])
        if label not in {0,product_id} or min(w,h)<=0:raise ValueError('Invalid source label')
        x=max(0.,min(640.,(cx-w/2)*640));y=max(0.,min(480.,(cy-h/2)*480))
        bw=max(0.,min(w*640,640-x));bh=max(0.,min(h*480,480-y))
        result['hand' if label==0 else 'product'].append([x,y,x+bw,y+bh])
    if len(result['hand'])>1 or len(result['product'])>1:raise ValueError('Multiple source objects')
    return result
def source_image(folder,stem):
    nested=folder/'images';folder=nested if nested.is_dir() else folder
    for ext in ['.jpg','.jpeg','.png','.JPG','.JPEG','.PNG']:
        p=folder/(stem+ext)
        if p.is_file():return p
    raise FileNotFoundError(str(folder/stem))


def prepare(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    bundle=Bundle();cfg=bundle.settings;rt=Features(bundle);gates=Gates(bundle)
    head=bundle.load('presence');binary=bundle.load('nonhand');product_head=bundle.load('product')
    root=Path(args.product_root);images=Path(args.product_images);work=root/'annotation';work.mkdir(exist_ok=False)
    originals={};before={};raw={};frames={};lookup={};features=[];routes=[];decisions=[];started=time.perf_counter()
    debug={};pass1={}
    try:
        for cam in args.cams:
            debug[cam]={Path(r['image']).stem:r for r in read_lines(root/cam/'debug/detections.jsonl')}
            first=Path(args.pass1_root)/cam/'debug/detections.jsonl'
            pass1[cam]={Path(r['image']).stem:next((x['box_xyxy'] for x in r['final'] if x['class_id']==0),None) for r in read_lines(first)}
            for label in sorted((root/cam/'labels').glob('*.txt')):
                stem=label.stem
                if not stem.isdigit():raise ValueError('ANNOTATION requires numeric frame stems')
                key=args.key_prefix+'/'+cam+'/'+stem;d=debug[cam][stem];p=boxes(label.read_text(),args.product_class_id)
                path=source_image(images/cam,stem)
                old={'key':key,'image':str(path),'cam':cam,'stem':stem,'source':d['hand_source'],
                     'product_source':d['product_source'],**p};originals[key]=old
                raw_hand=next((x['box_xyxy'] for x in d['final'] if x['class_id']==0),None)
                raw[key]={'box':raw_hand,'wrapper_removed':bool(raw_hand and not p['hand'])}
                with Image.open(path) as opened:
                    im=opened.convert('RGB');filtered=gates.run(im,old['source'],p['hand'],p['product'],cam)
                    before[key]={**old,'hand':filtered['hand'],'product':filtered['product']}
                    decisions.append({'key':key,'gates':filtered['decisions']})
                    pool=[]
                    if filtered['hand']:pool.append(filtered['hand'][0])
                    if raw_hand and (not filtered['hand'] or iou(raw_hand,filtered['hand'][0])<.95):pool.append(raw_hand)
                    suspicious=not filtered['hand'] or min(filtered['hand'][0][2]-filtered['hand'][0][0],filtered['hand'][0][3]-filtered['hand'][0][1])<32
                    if suspicious:
                        if pass1[cam].get(stem) and (not filtered['hand'] or iou(pass1[cam][stem],filtered['hand'][0])<.95):pool.append(pass1[cam][stem])
                        pool.extend(c['box_xyxy'] for c in d['hand_candidates'][:3])
                    frames[key]=[]
                    for box in pool:
                        ident=token(key,box)
                        if ident in lookup:continue
                        f=rt.run(im,box)
                        with torch.inference_mode():a,b,q=torch.sigmoid(head(torch.from_numpy(f)[None]))[0].tolist()
                        confidence=max(1e-6,min(1-1e-6,math.sqrt(a*b)*(.5+.5*q)))
                        c={'id':ident,'box':list(box),'local':a,'context':b,'quality':q,'confidence':confidence,'feature_index':len(features)}
                        features.append(f);lookup[ident]=c;frames[key].append(c)
                    frames[key]=sorted(frames[key],key=lambda c:(-c['confidence'],c['id']))[:4]
                    if filtered['product']:
                        box=filtered['product'][0];f=rt.run(im,box)
                        with torch.inference_mode():prob=torch.sigmoid(product_head(torch.from_numpy(f)[None])).item()
                        if prob<cfg['product_route_probability'] and min(box[2]-box[0],box[3]-box[1])>=12:
                            routes.append({'key':key,'box':box,'old_probability':prob,'image':str(path),'cam':cam})
                if len(originals)%500==0:print(json.dumps({'phase':'ANNOTATION_RGB','frames':len(originals),'seconds':time.perf_counter()-started}),flush=True)
        restored={};actions=[]
        for key,p in before.items():
            value=p['hand'];action='KEEP_EXISTING';r=raw[key]
            if not value and r['wrapper_removed']:
                c=lookup[token(key,r['box'])]
                value,action=choose_restore({'key':key,'box':r['box'],'current':[]},[c['local'],c['context'],c['quality']],cfg['restore'],before)
            restored[key]={**p,'hand':value,'source':'verified_E018_restore' if value and not p['hand'] else p['source']}
            if value!=p['hand']:actions.append({'key':key,'action':action})
        features=np.asarray(features,np.float32);paths={}
        for cam in args.cams:
            keys=sorted(k for k in restored if restored[k]['cam']==cam)
            a,_=one_direction(keys,frames,features,cfg['temporal']['motion_weight'])
            b,_=one_direction(keys,frames,features,cfg['temporal']['motion_weight'],True)
            paths.update({k:{'forward':a[k],'backward':b[k]} for k in keys})
        chosen={}
        for key,p in restored.items():
            value,action,_=choose_hand(key,p['hand'],frames[key],paths[key],lookup,cfg['temporal'])
            if not p['hand'] and value and neighbours(key,value[0],restored,radius=12)<2:value=[]
            chosen[key]={**p,'hand':value,'source':'verified_E020_restore' if value and not p['hand'] else p['source']}
            if value!=p['hand']:actions.append({'key':key,'action':action})
        for key,p in chosen.items():
            if not p['hand']:continue
            box=p['hand'][0];ident=token(key,box)
            if ident in lookup:f=features[lookup[ident]['feature_index']]
            else:
                with Image.open(p['image']) as opened:f=rt.run(opened.convert('RGB'),box)
            with torch.inference_mode():score=torch.softmax(binary(torch.from_numpy(f)[None]),1)[0,1].item()
            if min(box[2]-box[0],box[3]-box[1])>=cfg['nonhand']['minimum_native_side'] and score>=cfg['nonhand']['thresholds']['1']:
                p['hand']=[];actions.append({'key':key,'action':'E023_CONFIRMED_NONTARGET','score':score})
        selected=[]
        for cam in args.cams:selected+=sample([r for r in routes if r['cam']==cam],cfg['product_routes_per_camera'])
        write_lines(work/'raw_predictions.jsonl',originals.values());write_lines(work/'prepared_predictions.jsonl',chosen.values())
        write_lines(work/'gate_decisions.jsonl',decisions);write_lines(work/'actions.jsonl',actions)
        (work/'local_product_routes.json').write_text(json.dumps(selected));(work/'prepared.json').write_text(json.dumps({'release':bundle.manifest['release'],
            'frames':len(chosen),'feature_ROIs':len(features),'local_product_routes':len(selected),'seconds':time.perf_counter()-started,'GT_read':False}))
        print(json.dumps({'phase':'ANNOTATION_PREPARED','frames':len(chosen),'local_product_routes':len(selected)}),flush=True)
    finally:gates.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--product-root',required=True);p.add_argument('--pass1-root',required=True)
    p.add_argument('--product-images',required=True);p.add_argument('--product-class-id',required=True,type=int)
    p.add_argument('--key-prefix',required=True);p.add_argument('--cams',nargs='+',required=True);prepare(p.parse_args())
