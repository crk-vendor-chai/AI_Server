#!/usr/bin/env python3
"""Production-compatible ANNOTATION orchestrator. Old AS-IS remains an unchanged rollback entry."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,tempfile,time
from datetime import datetime, timezone
from pathlib import Path
from annotation_pipeline.bundle import Bundle

SCRIPT_DIR=Path(__file__).resolve().parent
PROJECT_ROOT=SCRIPT_DIR.parents[1]


def create_work_dir(runs: Path, run_name: str) -> Path:
    """Atomically reserve a timestamped workspace for one runner invocation."""
    runs.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    return Path(tempfile.mkdtemp(prefix=f'{run_name}__annotation_{timestamp}_', dir=runs))


def run(command,log_path,env):
    log_path.parent.mkdir(parents=True,exist_ok=True)
    print(json.dumps({'phase':'ANNOTATION_SUBPROCESS','log':str(log_path),'entry':Path(command[1]).name if len(command)>1 else ''}),flush=True)
    with log_path.open('x') as log:
        subprocess.run(command,cwd=PROJECT_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--img-root',default=str(PROJECT_ROOT/'img_download'));p.add_argument('--runs-root',default=str(PROJECT_ROOT/'runs/sam3_with_dino'))
    p.add_argument('--product',action='append',default=[]);p.add_argument('--cams',nargs='+',default=['cam_0','cam_2'])
    p.add_argument('--run-name',default=time.strftime('annotation_%Y%m%d_%H%M%S'));p.add_argument('--product-class-id',type=int)
    p.add_argument('--product-name',default='');p.add_argument('--root-folder',default='');p.add_argument('--product-idx',default='')
    p.add_argument('--python-bin',default=sys.executable);p.add_argument('--gpu',default='0');p.add_argument('--backend',default='video')
    p.add_argument('--dino-model',default='IDEA-Research/grounding-dino-base');p.add_argument('--skip-export',action='store_true')
    args,unknown=p.parse_known_args(argv)
    if args.backend!='video':raise ValueError('ANNOTATION release requires video AS-IS')
    if any(cam not in {'cam_0','cam_2'} for cam in args.cams):raise ValueError('ANNOTATION release has references for cam_0/cam_2 only')
    bundle=Bundle();runs=Path(args.runs_root).resolve();images=Path(args.img_root).resolve()
    products=[Path(v) if Path(v).is_absolute() else images/v for v in args.product] if args.product else sorted(v for v in images.iterdir() if v.is_dir() and not v.name.startswith('_'))
    if not products:raise ValueError('No product images')
    for product in products:
        if (runs/args.run_name/product.name).exists():raise FileExistsError('Refuse to overwrite an existing annotation run')
    outer=create_work_dir(runs,args.run_name)
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONUNBUFFERED']='1'
    if args.gpu:env['CUDA_VISIBLE_DEVICES']=args.gpu
    env.setdefault('TORCH_COMPILE_DISABLE','1')
    # Preserve the original full CLI; suppress only intermediate web export.
    base_command=[args.python_bin,str(SCRIPT_DIR/'asis_pipeline.py'),*argv,'--asis-work-dir',str(outer/'asis')]
    if '--skip-export' not in argv:base_command.append('--skip-export')
    run(base_command,outer/'asis.log',env)
    cpu_env=env.copy();cpu_env['CUDA_VISIBLE_DEVICES']=''
    vendor=os.environ.get('ANNOTATION_CPU_VENDOR','/opt/annotation_cpu_vendor')
    cpu_env['PYTHONPATH']=os.pathsep.join([str(SCRIPT_DIR),vendor,env.get('PYTHONPATH','')]);cpu_env.setdefault('OMP_NUM_THREADS','4')
    gpu_env=env.copy();gpu_env['PYTHONPATH']=os.pathsep.join([str(SCRIPT_DIR),env.get('PYTHONPATH','')])
    for product in products:
        product_root=runs/args.run_name/product.name;first=runs/(args.run_name+'__pass1')/product.name
        cams=[cam for cam in args.cams if (product/cam).is_dir()]
        if not cams:raise ValueError('No supported cameras')
        class_id=args.product_class_id or int(product.name.split('_',1)[0])
        if class_id<=0:raise ValueError('Product class must not collide with hand class 0')
        common=['--product-root',str(product_root)]
        run([args.python_bin,'-B','-m','annotation_pipeline.postprocess',*common,'--pass1-root',str(first),
             '--product-images',str(product),'--product-class-id',str(class_id),'--key-prefix',product.name.split('_',1)[0],'--cams',*cams],outer/(product.name+'_cpu.log'),cpu_env)
        run([args.python_bin,'-B','-m','annotation_pipeline.local_product',*common,'--product-name',product.name,
             '--dino-model',args.dino_model],outer/(product.name+'_local_dino.log'),gpu_env)
        run([args.python_bin,'-B','-m','annotation_pipeline.finalize',*common,'--product-class-id',str(class_id),'--cams',*cams],outer/(product.name+'_finalize.log'),cpu_env)
        if not args.skip_export:
            command=[args.python_bin,str(SCRIPT_DIR/'export_web_annotations.py'),'--results-root',str(product_root),
                     '--image-root',str(product),'--cams',*cams,'--product-class-id',str(class_id)]
            for option,value in [('--product-name',args.product_name),('--root-folder',args.root_folder),('--product-idx',args.product_idx)]:
                if value:command.extend([option,value])
            run(command,product_root/'out/export_ANNOTATION.log',cpu_env)
            files=list((product_root/'out').glob('annotations_*.json'))
            if len(files)!=1:raise RuntimeError('Expected exactly one final web-review JSON')
            payload=json.loads(files[0].read_text())
            if {c['id'] for c in payload['categories']}!={1,2}:raise RuntimeError('Unexpected COCO categories')
        result={'release':bundle.manifest['release'],'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'model_manifest_sha256':hashlib.sha256((bundle.folder/'manifest.json').read_bytes()).hexdigest(),
            'GPU':args.gpu,'GT_used':False,'MinIO_write_by_runner':False,'web_export':not args.skip_export}
        with (product_root/'ANNOTATION_COMPLETE.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'phase':'ANNOTATION_COMPLETE','release':bundle.manifest['release'],'products':len(products)}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
