"""Evaluate a local GQDet EMA checkpoint with native postprocessing."""
import argparse
import json
import os
from pathlib import Path
import sys

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['hrsid','air'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--data-root',type=Path)
    p.add_argument('--check-only',action='store_true',help='Strict checkpoint loading; no inference')
    args=p.parse_args()
    checkpoint=args.checkpoint.resolve(); output=args.output.resolve()
    root=Path(__file__).resolve().parent
    repo=root
    data=args.data_root.resolve() if args.data_root else root/'datasets'/args.dataset
    os.chdir(repo);sys.path.insert(0,str(repo))
    import torch
    from src.core import YAMLConfig
    torch.set_num_threads(4)
    cfg=YAMLConfig(str(repo/'configs'/f'gqdet_{args.dataset}.yml'),resume=None,tuning=None)
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    cfg.yaml_cfg['val_dataloader']['num_workers']=0
    cfg.yaml_cfg['val_dataloader']['dataset']['img_folder']=str(data/'images'/('test' if args.dataset=='hrsid' else ''))
    cfg.yaml_cfg['val_dataloader']['dataset']['ann_file']=str(data/'annotations/instances_test.json')
    # Only load local trusted checkpoints; resumable training states can contain Python RNG state.
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    weights=state['ema']['module']
    model=cfg.model.float().eval()
    model.load_state_dict(weights,strict=True)
    result={'dataset':args.dataset,'checkpoint':checkpoint.name,'epoch_index':state.get('last_epoch'),
            'strict_load':True,'tensor_count':len(weights)}
    if not args.check_only:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        model=model.to(args.device);post=cfg.postprocessor.to(args.device).eval()
        loader=cfg.val_dataloader
        rows=[]
        category_ids=sorted(loader.dataset.coco.getCatIds())
        assert len(category_ids)==1
        with torch.no_grad():
            for samples,targets in loader:
                outputs=model(samples.to(args.device))
                sizes=torch.stack([t['orig_size'] for t in targets]).to(args.device)
                predictions=post(outputs,sizes)
                for t,r in zip(targets,predictions):
                    assert len(r['scores'])==300
                    for b,s in zip(r['boxes'].cpu().tolist(),r['scores'].cpu().tolist()):
                        rows.append({'image_id':int(t['image_id'].item()),'category_id':category_ids[0],
                                     'bbox':[b[0],b[1],b[2]-b[0],b[3]-b[1]],'score':s})
        gt=COCO(str(data/'annotations/instances_test.json'))
        ev=COCOeval(gt,gt.loadRes(rows),'bbox');ev.params.maxDets=[1,10,100]
        ev.evaluate();ev.accumulate();ev.summarize()
        result['metrics_percent']=dict(zip(['AP50_95','AP50','AP75','AP_S','AP_M','AP_L','AR1','AR10','AR100','AR_S','AR_M','AR_L'],[float(x)*100 for x in ev.stats]))
        result['prediction_count']=len(rows)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.with_suffix('.predictions.json').write_text(json.dumps(rows),encoding='utf8')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
