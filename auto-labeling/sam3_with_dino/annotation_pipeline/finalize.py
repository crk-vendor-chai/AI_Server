"""Synchronize final detection labels, valid raw masks, debug and visualization before COCO."""
import argparse,hashlib,json,math,os,shutil
from pathlib import Path
from PIL import Image,ImageDraw
from .bundle import Bundle
from .geometry import iou
from .postprocess import read_lines


def atomic(path,text):
    temporary=path.with_name(path.name+'.annotation-tmp')
    with temporary.open('x') as f:f.write(text)
    os.replace(temporary,path)
def main(args):
    bundle=Bundle();root=Path(args.product_root);work=root/'annotation'
    rows=read_lines(work/'predictions.jsonl');old={r['key']:r for r in read_lines(work/'raw_predictions.jsonl')}
    raw_folder=work/'raw_labels';raw_folder.mkdir(exist_ok=False)
    changed={'hand':0,'product':0};mask_drops=0
    for cam in args.cams:
        shutil.copytree(root/cam/'labels',raw_folder/cam/'labels')
        if (root/cam/'labels_seg').is_dir():shutil.copytree(root/cam/'labels_seg',raw_folder/cam/'labels_seg')
        debug_path=root/cam/'debug/detections.jsonl';debug=read_lines(debug_path);by_stem={Path(d['image']).stem:d for d in debug}
        for row in (r for r in rows if r['cam']==cam):
            key=row['key'];stem=row['stem'];d=by_stem[stem];original_final=d['final'];label=[]
            for role,class_id in [('hand',0),('product',args.product_class_id)]:
                if len(row[role])>1:raise ValueError('Multiple output objects')
                changed[role]+=row[role]!=old[key][role]
                for box in row[role]:
                    if len(box)!=4 or not all(math.isfinite(v) for v in box) or box[2]<=box[0] or box[3]<=box[1]:raise ValueError('Invalid ANNOTATION box')
                    x1,y1,x2,y2=box
                    values=[(x1+x2)/1280,(y1+y2)/960,(x2-x1)/640,(y2-y1)/480]
                    if min(values[2:])<=0:raise ValueError('Collapsed output box')
                    label.append(str(class_id)+' '+' '.join(f'{v:.9f}' for v in values))
            atomic(root/cam/'labels'/(stem+'.txt'),'\n'.join(label)+('\n' if label else ''))
            seg=root/cam/'labels_seg'/(stem+'.txt')
            if seg.exists():
                keep=[]
                for line in seg.read_text().splitlines():
                    if not line.strip():continue
                    class_id=int(line.split()[0]);role='hand' if class_id==0 else 'product'
                    raw_box=next((v['box_xyxy'] for v in original_final if v['class_id']==class_id),None)
                    if row[role] and raw_box and iou(row[role][0],raw_box)>=.9999:keep.append(line)
                    else:mask_drops+=1
                atomic(seg,'\n'.join(keep)+('\n' if keep else ''))
            d['raw_before_annotation']={'final':original_final,'hand_source':d.get('hand_source'),'product_source':d.get('product_source')}
            d['final']=[]
            for role,class_id in [('hand',0),('product',args.product_class_id)]:
                for box in row[role]:
                    prior=next((v for v in original_final if v['class_id']==class_id),{})
                    d['final'].append({**prior,'class_id':class_id,'box_xyxy':box,'annotation_role':role,'score_provenance':'raw_score_if_present_not_recalibrated'})
            d['hand_source']=row['source'] if row['hand'] else 'none'
            d['annotation_pipeline']={'release':bundle.manifest['release'],'hand':row['hand'],'product':row['product'],'GT_used':False}
            with Image.open(row['image']) as opened:im=opened.convert('RGB')
            draw=ImageDraw.Draw(im)
            for role,color in [('hand','#ff4444'),('product','#33dd66')]:
                for b in row[role]:draw.rectangle(b,outline=color,width=3)
            im.save(root/cam/'visualized'/(stem+'.jpg'),quality=85)
        shutil.copy2(debug_path,raw_folder/cam/'detections.jsonl')
        atomic(debug_path,''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in debug))
    result={'release':bundle.manifest['release'],'frames':len(rows),'changed':changed,'stale_masks_removed':mask_drops,
            'manifest_sha256':hashlib.sha256((bundle.folder/'manifest.json').read_bytes()).hexdigest(),'max_hands':1,'GT_used':False}
    with (work/'finalized.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'phase':'ANNOTATION_FINALIZED',**result}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--product-root',required=True);p.add_argument('--product-class-id',required=True,type=int)
    p.add_argument('--cams',required=True,nargs='+');main(p.parse_args())
