"""E024: two fresh DINO crops, frozen image evidence, replacement only (never deletion)."""
import argparse,json,math,sys,time
from pathlib import Path
import torch
from PIL import Image
from .bundle import Bundle
from .gates import Features
from .geometry import *
from .temporal import choose_product
from .postprocess import read_lines,write_lines


def main(args):
    bundle=Bundle();root=Path(args.product_root);work=root/'annotation'
    rows={r['key']:r for r in read_lines(work/'prepared_predictions.jsonl')}
    routes=json.loads((work/'local_product_routes.json').read_text());decisions=[];started=time.perf_counter()
    if routes:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        import auto_annotate as engine
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('E024 requires exactly one assigned GPU')
        engine.set_dino_autocast(False)
        processor,model=engine.load_grounding_dino(args.dino_model,'cuda')
        rt=Features(bundle);head=bundle.load('product')
        for route in routes:
            views=[]
            with Image.open(route['image']) as opened:
                im=opened.convert('RGB')
                for scale in [1.4,2.]:
                    region=expand(route['box'],scale,128);region=[math.floor(region[0]),math.floor(region[1]),math.ceil(region[2]),math.ceil(region[3])]
                    detected=engine.detect_with_dino(im.crop(region),processor,model,'cuda',engine.product_prompts_from_folder(args.product_name),1,'product',.18,.20,5)
                    view=[]
                    for c in detected:
                        box=clip([c.box_xyxy[0]+region[0],c.box_xyxy[1]+region[1],c.box_xyxy[2]+region[0],c.box_xyxy[3]+region[1]])
                        f=rt.run(im,box)
                        with torch.inference_mode():prob=torch.sigmoid(head(torch.from_numpy(f)[None])).item()
                        view.append({'box':box,'dino_score':c.score,'product_probability':prob})
                    views.append(view)
            candidate={**route,'old_box':route['box'],'views':views}
            selected=choose_product(candidate,bundle.settings['local_product'])
            rows[route['key']]['product']=[selected]
            decisions.append({**candidate,'selected':selected,'changed':selected!=route['box']})
        del model;torch.cuda.empty_cache()
    write_lines(work/'predictions.jsonl',rows.values());write_lines(work/'local_product_decisions.jsonl',decisions)
    (work/'local_product_complete.json').write_text(json.dumps({'routes':len(routes),'seconds':time.perf_counter()-started,'GT_used':False}))
    print(json.dumps({'phase':'E024_COMPLETE','routes':len(routes),'replacements':sum(d['changed'] for d in decisions)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--product-root',required=True);p.add_argument('--product-name',required=True)
    p.add_argument('--dino-model',default='IDEA-Research/grounding-dino-base');main(p.parse_args())
