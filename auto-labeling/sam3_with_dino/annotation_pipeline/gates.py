"""Frozen E006/E011/E014/E015 gates and shared local/context features."""
import math,os
import numpy as np
import torch
from PIL import Image
from . import local_cpu,spatial,native,sticker
from .geometry import expand,inter
from .patches import patch_boxes


class Features:
    def __init__(self,bundle):self.encoder=bundle.load('pooled')
    def run(self,im,box):
        local=local_cpu.feature(self.encoder,im,box)
        context=local_cpu.feature(self.encoder,im,expand(box,1.6,80))
        result=np.concatenate([local,context]).astype(np.float32)
        if result.shape!=(1152,) or not np.isfinite(result).all():raise ValueError('Invalid ANNOTATION feature')
        return result


class Gates:
    def __init__(self,bundle,use_reference_background=True):
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('ANNOTATION gate worker must be CPU-only')
        from .fallback import FallbackVerifier
        local_cpu.configure(4);self.bundle=bundle;self.dino=FallbackVerifier(bundle)
        self.encoder=bundle.load('spatial');self.patch=bundle.load('patch');self.cnn=bundle.load('sticker')
        self.bank=np.load(bundle.path('bank'));self.references={};self.foreground={}
        self.use_reference_background=use_reference_background
        for cam,cfg in (bundle.settings['references'].items() if use_reference_background else []):
            with Image.open(bundle.path(cfg['asset'])) as im:self.references[cam]=im.convert('RGB')
            self.foreground[cam]=cfg['foreground']
    def patch_keep(self,im,box):
        cfg=self.bundle.settings['patch'];whole=None;maximum=0.;calls=0
        with torch.inference_mode():
            for b in patch_boxes(box):
                ctx=spatial.context(b);support=spatial.grid_mask(b,ctx)*spatial.grid_mask(ctx,ctx)
                if not support.any():raise ValueError('No ROI support cells')
                z=self.encoder(spatial.tensor(im,b).unsqueeze(0))[0]
                value=torch.sigmoid(spatial.presence_logit(self.patch(z),support.unsqueeze(0))).item()
                calls+=1;maximum=max(maximum,value)
                if whole is None:
                    whole=value
                    if whole>=cfg['whole_threshold']:return {'keep':True,'whole':whole,'maximum_seen':maximum,'calls':calls}
                    if whole<cfg['whole_threshold']*cfg['minimum_whole_fraction']:return {'keep':False,'whole':whole,'maximum_seen':maximum,'calls':calls}
                elif value>=cfg['strong_patch_threshold']:return {'keep':True,'whole':whole,'maximum_seen':maximum,'calls':calls}
        return {'keep':False,'whole':whole,'maximum_seen':maximum,'calls':calls}
    def sticker_score(self,im,box):
        with torch.inference_mode():return torch.sigmoid(self.cnn(sticker.tensor(im,box).unsqueeze(0))).item()
    def sticker_check(self,im,box,cam):
        if not self.use_reference_background:
            return {'reject':False,'current_score':None,'guard':None,'reason':'top_view_without_calibrated_reference'}
        cfg=self.bundle.settings['sticker'];current=self.sticker_score(im,box);e=None
        if cfg['enabled'] and current>=cfg['current_min']:
            b=spatial.clip(box)
            if any(inter(b,g)>0 for g in self.foreground[cam]):
                e={'supported':False,'reference_foreground':True,'reference_score':None,'correlation':None}
            else:
                value=self.sticker_score(self.references[cam],box)
                a=np.asarray(im.crop(b).convert('L').resize((64,64),Image.Resampling.BILINEAR),dtype=np.float64).ravel()
                z=np.asarray(self.references[cam].crop(b).convert('L').resize((64,64),Image.Resampling.BILINEAR),dtype=np.float64).ravel()
                a-=a.mean();z-=z.mean();den=float(np.linalg.norm(a)*np.linalg.norm(z))
                e={'supported':True,'reference_foreground':False,'reference_score':value,'correlation':float(np.dot(a,z)/den) if den>1e-8 else -1.}
        reject=bool(cfg['enabled'] and current>=cfg['current_min'] and e is not None and e['supported'] and e['reference_score']>=cfg['reference_min'] and e['correlation']>=cfg['NCC_min'])
        return {'reject':reject,'current_score':current,'guard':e}
    def run(self,im,source,hand,product,cam):
        if im.size!=(640,480) or (self.use_reference_background and cam not in self.references):raise ValueError('Unsupported ANNOTATION image size or camera')
        if len(hand)>1 or len(product)>1:raise ValueError('ANNOTATION outputs at most one hand/product')
        result={'hand':hand,'product':product,'decisions':{}}
        if hand:
            if source=='dino_fallback':
                d=self.dino.verify(im,source,hand)
                if d['reason']=='runtime_error_preserved':raise RuntimeError('E006 gate error: '+d.get('error','unknown'))
                result['decisions']['E006']=d;result['hand']=d['hand']
                if result['hand']:
                    d=self.patch_keep(im,hand[0]);result['decisions']['E011']=d
                    if not d['keep']:result['hand']=[]
            elif source and source.startswith('sam3'):
                b=hand[0];w,h=b[2]-b[0],b[3]-b[1];cfg=self.bundle.settings['native']
                reliable=min(w,h)>=cfg['minimum_original_short_side'] and w*h>=cfg['minimum_original_area'];reject=False
                if reliable and cfg['enabled']:
                    with torch.inference_mode():f=self.encoder(native.tensor(im,b).unsqueeze(0))[1].squeeze(0).numpy()
                    if not np.isfinite(f).all():raise ValueError('Invalid SAM3 feature')
                    reject=native.matches(f,self.bank['features'],self.bank['thresholds'])
                result['decisions']['E014']={'reject':reject,'original_pixels_reliable':reliable}
                if reject:result['hand']=[]
            else:raise ValueError('Unknown hand source: '+str(source))
        if product:
            d=self.sticker_check(im,product[0],cam);result['decisions']['E015']=d
            if d['reject']:result['product']=[]
        return result
    def close(self):self.dino.close()
