import math
import numpy as np
from .geometry import *

def one_direction(keys,frames,features,motion_weight,reverse=False):
    """No GT argument. Only confident target hypotheses enter branch memory."""
    keys=list(reversed(keys)) if reverse else keys
    states=[{'score':0.,'last':None,'memory':[],'history':[]}];result={};windows=0
    for start in range(0,len(keys),48):
        chunk=keys[start:start+48];windows+=1
        for key in chunk:
            candidates=frames[key];maximum=max([c['confidence'] for c in candidates]+[0.]);branches=[]
            for state in states:
                # An explicit absent state prevents forced tracking through occlusion/background.
                branches.append({'score':state['score']+math.log(max(1e-6,1-maximum))-(.03 if state['last'] else 0),
                    'last':None,'memory':state['memory'],'history':state['history']+[(key,None)]})
                for c in candidates:
                    transition=motion_weight*iou(c['box'],state['last']['box']) if state['last'] else 0.
                    memory=state['memory'];appearance=0.
                    if memory:
                        f=features[c['feature_index']][:576]
                        appearance=.15*max(float(np.dot(f,features[m['feature_index']][:576])) for m in memory)
                    if c['local']>=.95 and c['context']>=.95 and c['quality']>=.5:memory=[*memory,c][-3:]
                    branches.append({'score':state['score']+math.log(c['confidence'])+transition+appearance,
                        'last':c,'memory':memory,'history':state['history']+[(key,c['id'])]})
            branches.sort(key=lambda b:-b['score']);states=[];identities=set()
            for b in branches:
                # Retain different recent hypotheses instead of collapsing all beams to one.
                signature=tuple(x[1] for x in b['history'][-3:])
                if signature not in identities:states.append(b);identities.add(signature)
                if len(states)==3:break
        best=max(states,key=lambda s:s['score']);result.update(dict(best['history']))
        high=max(s['score'] for s in states)
        for s in states:s['score']-=high;s['history']=[]
    return result,windows

def neighbours(key,b,before,radius=12):
    prefix,stem=key.rsplit('/',1);frame=int(stem);count=0
    for direction in [-1,1]:
        for delta in range(1,radius+1):
            k=f'{prefix}/{frame+direction*delta:06d}';p=before.get(k)
            if p and p['hand']:
                a=p['hand'][0]
                if iou(a,b)>=.1:count+=1;break
    return count

def choose_restore(route,scores,config,before):
    """Inference consumes RGB-derived scores/geometry only, never GT."""
    if route['current']:return route['current'],'KEEP_EXISTING'
    a,b,q=scores
    if not config.get('enabled'):return [],'ABSTAIN_REVIEW'
    if a>=config['presence_min'] and b>=config['presence_min'] and q>=config['quality_min'] and neighbours(route['key'],route['box'],before)>=config['neighbours_min']:
        return [route['box']],'RESTORE_CONFIRMED_VISIBLE_HAND'
    if a<.05 and b<.05:return [],'REJECT_CONFIRMED_NONTARGET'
    return [],'ABSTAIN_REVIEW'

def choose_hand(key,current,candidates,path,lookup,param):
    if current:return current,'KEEP_EXISTING',None
    if not param.get('enabled'):return [],'ABSTAIN_REVIEW',None
    a,b=lookup.get(path['forward']),lookup.get(path['backward'])
    if a is None or b is None or iou(a['box'],b['box'])<param['path_agreement']:return [],'ABSTAIN_REVIEW',None
    valid=[]
    for c in candidates:
        if min(c['local'],c['context'])<param['presence_min'] or c['quality']<param['quality_min']:continue
        if min(iou(c['box'],a['box']),iou(c['box'],b['box']))<param['candidate_agreement']:continue
        valid.append(c)
    if not valid:return [],'ABSTAIN_REVIEW',None
    c=max(valid,key=lambda c:c['confidence']*math.sqrt(iou(c['box'],a['box'])*iou(c['box'],b['box'])))
    return [c['box']],'RECOVER_CONFIRMED_VISIBLE_HAND',c['id']

def choose_product(row,param):
    best=None;rank=-1.
    if not param.get('enabled'):return row['old_box']
    for a in row['views'][0]:
        for b in row['views'][1]:
            agreement=iou(a['box'],b['box']);prob=min(a['product_probability'],b['product_probability']);confidence=min(a['dino_score'],b['dino_score'])
            if agreement<param['agreement'] or prob<param['probability'] or confidence<param['dino_min'] or prob-row['old_probability']<.5:continue
            c=max([a,b],key=lambda c:c['dino_score']);value=agreement*prob*confidence
            if value>rank:best=c['box'];rank=value
    return best if best is not None else row['old_box']
