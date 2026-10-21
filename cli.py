"""Portable fresh-run CLI; see README.md for historical versus fresh runs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import core


def write(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def label(a,b,d,delta):
    N,L = a.shape; Q = delta.shape[1]; spectra = []; error = 0.; energy = 0.
    for k in range(1+L*Q):
        dd = d.copy()
        if k: dd[:,(k-1)//Q] += delta[:,(k-1)%Q]
        r,t = core.tmm(a,b,dd)
        error = max(error,float(abs(r-core.recursive_reflectance(a,b,dd)).max()))
        energy = max(energy,float(abs(r+t-1).max()))
        spectra.append(r)
    if error >= 1e-10 or energy >= 1e-10:
        raise AssertionError(f'Physics validation failed: solver={error}, energy={energy}')
    spectra = np.stack(spectra,1)
    u = np.stack((spectra[...,:15].mean(-1),spectra[...,26:].mean(-1)),-1)
    y = np.zeros((N,24,Q,2))
    y[:,:L] = (u-u[:,:1])[:,1:].reshape(N,L,Q,2)
    return y,dict(solver_max_error=error,energy_max_error=energy,endpoints_per_solver=N*(1+L*Q))


def generate(out, layers=(8,12,16), count=512, seed=2026101770200, train=True, domain='normal'):
    out = Path(out); out.mkdir(parents=True,exist_ok=False)
    rng = np.random.default_rng(seed); ledger = []
    for L in layers:
        a,b,d = core.sample(rng,L,count,domain)
        if train:
            small=rng.uniform(.003,.006,count); large=rng.uniform(.008,.012,count)
            delta=np.stack((-small,small,-large,large),1)
        else: delta=np.tile([-.01,.01],(count,1))
        y,cost = label(a,b,d,delta); ledger.append(dict(layers=L,**cost))
        np.savez_compressed(out/f'L{L}.npz',a=a,b=b,d=d,delta=delta,truth=y,lengths=np.full(count,L))
    write(out/'generation.json',dict(mode='fresh_reproducible_experiment',training=train,seed=seed,
          domain=domain,layers=list(layers),objects_per_length=count,physics=ledger))
    return ledger


def load_data(folder, arm, unit=None):
    paths=sorted(Path(folder).glob('*.npz'))
    if not paths: raise ValueError(f'No NPZ inputs in {folder}')
    sources=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as z:
            if not {'a','b','d','delta','truth'}.issubset(z.files): continue
            sources.append({k:z[k] for k in ('a','b','d','delta','truth')})
    if not sources: raise ValueError('Need a,b,d,delta,truth in each input archive')
    if unit is None: unit=float(np.concatenate([s['d'].ravel() for s in sources]).mean())
    blocks=[]
    for src in sources:
        a,b,d,delta,y=[src[k] for k in ('a','b','d','delta','truth')]
        N,L=a.shape
        if y.shape != (N,24,delta.shape[1],2): raise ValueError('Incorrect padded truth shape')
        ar,br,dr=[v[:,::-1].copy() for v in (a,b,d)]
        mode=core.feature_kind(arm)
        blocks.append(dict(F=core.features(a,b,d,mode),FR=core.features(ar,br,dr,mode),
            Q=core.query(a,b,delta),QR=core.query(ar,br,delta),
            P=core.positions(a,b,d,arm,unit),PR=core.positions(ar,br,dr,arm,unit),
            Y=y,lengths=np.full(N,L)))
    return {k:np.concatenate([s[k] for s in blocks]) for k in blocks[0]},unit


def normalization(data):
    mask=np.arange(24)[None]<data['lengths'][:,None]
    f=data['F'][mask]; q=data['Q'][mask].reshape(-1,7)
    return dict(fm=f.mean(0),fs=np.maximum(f.std(0),1e-6),qm=q.mean(0),qs=np.maximum(q.std(0),1e-6))


def train(data_dir, out, arm='full_aug', seed=0, steps=800, batch=64, device='cpu', augmentation=True):
    import torch
    torch.set_num_threads(2); torch.backends.mha.set_fastpath_enabled(False)
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    data,unit=load_data(data_dir,arm); norm=normalization(data)
    lengths=data['lengths']; mask=np.arange(24)[None]<lengths[:,None]
    scale=np.maximum(np.sqrt((data['Y']**2).sum((0,1,2))/(mask.sum()*data['Y'].shape[2])),1e-6)
    tf=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
    X={k:tf((data[k]-norm['fm'])/norm['fs']) for k in ('F','FR')}
    Q={k:tf((data[k]-norm['qm'])/norm['qs']) for k in ('Q','QR')}
    P={k:tf(data[k]) for k in ('P','PR')}
    lens=torch.as_tensor(lengths,device=device); mm=torch.as_tensor(mask,device=device)
    target=tf(data['Y']/scale); qzero=tf(-norm['qm']/norm['qs'])
    net=core.make_model(arm,seed,device); opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    rng=np.random.default_rng(20261017711+seed); losses=[]; net.train()
    for step in range(steps):
        ids=rng.integers(len(lengths),size=batch); bm=mm[ids]
        reverse_key='FR' if augmentation else 'F'
        query_key='QR' if augmentation else 'Q'
        position_key='PR' if augmentation else 'P'
        pred=net(torch.cat((X['F'][ids],X[reverse_key][ids])),torch.cat((Q['Q'][ids],Q[query_key][ids])),
                 qzero,torch.cat((bm,bm)),torch.cat((P['P'][ids],P[position_key][ids])))
        p=pred[:batch]
        r=core.torch_reverse(torch,pred[batch:],lens[ids]) if augmentation else pred[batch:]
        loss=(((p-target[ids]).square()+(r-target[ids]).square())*.5*bm[:,:,None,None]).sum()/(bm.sum()*target.shape[2]*2)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    torch.save(net.state_dict(),out/'model.pt')
    np.savez(out/'normalization.npz',**norm,scale=scale)
    for filename in ('core.py','cli.py','requirements.txt','README.md'):
        shutil.copy2(Path(__file__).resolve().parent/filename,out/filename)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(data_dir).glob('*.npz')}
    config=dict(arm=arm,seed=seed,steps=steps,batch=batch,lr=.001,weight_decay=.01,
        position_unit_um=unit,parameters=sum(p.numel() for p in net.parameters()),
        torch_version=torch.__version__,numpy_version=np.__version__,input_sha256=hashes,
        training_objects=len(lengths),training_object_forwards=steps*batch*2,
        augmentation=bool(augmentation),
        training_policy='original-only normalization; shared query zero anchor; '+('reverse two-view loss' if augmentation else 'duplicated original two-view loss'),loss=losses)
    write(out/'config.json',config)
    return config


def evaluate(data_dir, checkpoint, out, device='cpu'):
    import torch
    torch.set_num_threads(2); torch.backends.mha.set_fastpath_enabled(False)
    checkpoint=Path(checkpoint); config=json.loads((checkpoint/'config.json').read_text())
    data,_=load_data(data_dir,config['arm'],config['position_unit_um'])
    with np.load(checkpoint/'normalization.npz') as z: norm=dict(z)
    net=core.make_model(config['arm'],config['seed'],device)
    net.load_state_dict(torch.load(checkpoint/'model.pt',map_location=device,weights_only=True)); net.eval()
    tf=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
    preds=[]; zeros=[]
    with torch.no_grad():
        for lo in range(0,len(data['lengths']),64):
            sl=slice(lo,lo+64); mask=torch.arange(24,device=device)[None]<torch.as_tensor(data['lengths'][sl],device=device)[:,None]
            x=tf((data['F'][sl]-norm['fm'])/norm['fs']); q=tf((data['Q'][sl]-norm['qm'])/norm['qs'])
            q0=tf(-norm['qm']/norm['qs']); pos=tf(data['P'][sl])
            preds.append(net(x,q,q0,mask,pos).cpu().numpy()*norm['scale'])
            zeros.append(float(net(x,q0.expand(len(x),24,1,7),q0,mask,pos).abs().max()))
    pred=np.concatenate(preds); met=core.metrics(data['Y'],pred,data['lengths'])
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'predictions.npz',prediction=pred,lengths=data['lengths'])
    np.savez_compressed(out/'object_metrics.npz',**met)
    summary={k:float(np.mean(met[k])) for k in ('regret','layer_regret','within_layer_regret','noop','gain_rmse')}
    summary['zero_query_max_error']=max(zeros)
    write(out/'summary.json',summary)
    return summary


def smoke(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    ledger=generate(out/'train',(3,5),3,seed=19)
    ledger+=generate(out/'test',(4,),2,seed=23,train=False)
    config=train(out/'train',out/'checkpoint',steps=2,batch=2)
    result=evaluate(out/'test',out/'checkpoint',out/'evaluation')
    repeat=evaluate(out/'test',out/'checkpoint',out/'evaluation_reload')
    with np.load(out/'evaluation'/'predictions.npz') as a, np.load(out/'evaluation_reload'/'predictions.npz') as b:
        reload_error=float(abs(a['prediction']-b['prediction']).max())
    assert reload_error==0 and result['zero_query_max_error']<1e-6
    report=dict(status='PASS',purpose='CPU software smoke only; not scientific evidence',optimizer_steps=config['steps'],
        physics_solver_max_error=max(x['solver_max_error'] for x in ledger),
        physics_energy_max_error=max(x['energy_max_error'] for x in ledger),reload_prediction_max_error=reload_error,
        zero_query_max_error=result['zero_query_max_error'])
    write(out/'smoke.json',report); return report


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate'); g.add_argument('--out',required=True); g.add_argument('--layers',type=int,nargs='+',default=[8,12,16])
    g.add_argument('--count',type=int,default=512); g.add_argument('--seed',type=int,default=2026101770200)
    g.add_argument('--evaluation',action='store_true'); g.add_argument('--domain',choices=['normal','high','dispersion','strong'],default='normal')
    t=sub.add_parser('train'); t.add_argument('--data',required=True); t.add_argument('--out',required=True)
    t.add_argument('--arm',choices=core.ARMS,default='full_aug'); t.add_argument('--seed',type=int,default=0)
    t.add_argument('--steps',type=int,default=800); t.add_argument('--batch',type=int,default=64); t.add_argument('--device',default='cpu')
    t.add_argument('--no-reversal',action='store_true',help='Duplicate original view instead of reverse augmentation, preserving forward count')
    e=sub.add_parser('evaluate'); e.add_argument('--data',required=True); e.add_argument('--checkpoint',required=True)
    e.add_argument('--out',required=True); e.add_argument('--device',default='cpu')
    s=sub.add_parser('smoke'); s.add_argument('--out',required=True)
    a=p.parse_args()
    if a.command=='generate': result=generate(a.out,a.layers,a.count,a.seed,not a.evaluation,a.domain)
    elif a.command=='train': result=train(a.data,a.out,a.arm,a.seed,a.steps,a.batch,a.device,not a.no_reversal)
    elif a.command=='evaluate': result=evaluate(a.data,a.checkpoint,a.out,a.device)
    else: result=smoke(a.out)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
