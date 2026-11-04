"""Dataset-specific final training settings in global epoch coordinates."""
from ..misc import dist_utils


def matching_class_cost(config, epoch, default):
    value=float(default)
    for item in sorted(config.get('matching_class_cost_overrides',[]),key=lambda x:x['start_epoch']):
        if epoch>=int(item['start_epoch']):value=float(item['value'])
    return value


def apply_runtime_schedule(solver, epoch):
    entries=sorted(solver.cfg.yaml_cfg.get('runtime_schedule',[]),key=lambda x:x['start_epoch'])
    active=None
    for index,item in enumerate(entries):
        if epoch>=int(item['start_epoch']):active=(index,item)
    if active is None or getattr(solver,'_final_runtime_index',None)==active[0]:return
    index,item=active
    cfg=solver.cfg
    for kind,field in [('train','train_workers'),('val','val_workers')]:
        if field not in item:continue
        key=kind+'_dataloader'
        cfg.yaml_cfg[key]['num_workers']=int(item[field])
        setattr(cfg,'_'+key,None)
        loader=getattr(cfg,key)
        setattr(solver,key,dist_utils.warp_loader(loader,shuffle=loader.shuffle))
    if 'seed' in item:dist_utils.setup_seed(int(item['seed']))
    solver._final_runtime_index=index
