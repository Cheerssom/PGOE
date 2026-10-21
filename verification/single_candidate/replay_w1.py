"""NumPy-only replay of fixed frozen proposals and selected-candidate TMM gate.

Run from any directory: python /path/to/W1_verification_20261021/replay_w1.py
This replays scientific quality, not the machine-dependent CPU timing benchmark.
"""
from pathlib import Path
import json
import numpy as np
import core
ROOT=Path(__file__).resolve().parent
def utility(ref):return np.stack((ref[...,:15].mean(-1),ref[...,26:].mean(-1)),-1)
def evaluate(truth,choice):
 a=np.concatenate((np.zeros((len(truth),1,2)),truth.reshape(len(truth),-1,2)),1)
 gain=np.take_along_axis(a,choice[:,None,:],1)[:,0];best=a.max(1)
 return dict(regret=float((best-gain).mean()),negative=float((gain < -1e-10).mean()),gainrecovery=float(gain.sum()/best.sum()))
def main():
 expected=json.loads((ROOT/'evidence.json').read_text(encoding='utf8'))
 out={}
 for dom in ['depth20','depth24','strong16']:
  with np.load(ROOT/'data'/f'{dom}.npz') as z:a,b,d,truth,pred=[z[k] for k in ['a','b','d','truth','predictions']]
  baseline=utility(core.tmm(a,b,d)[0]);after=[];before=[]
  for seed,p in enumerate(pred):
   choice=np.concatenate((np.zeros((len(p),1,2)),p.reshape(len(p),-1,2)),1).argmax(1)
   pairs=sorted({(i,int(k)) for i in range(len(a)) for k in choice[i] if k>0})
   queried=np.zeros_like(choice,dtype=float)
   if pairs:
    ii=np.array([i for i,k in pairs]);kk=np.array([k for i,k in pairs]);dd=d[ii].copy()
    dd[np.arange(len(ii)),(kk-1)//2]+=np.where((kk-1)%2==0,-.01,.01)
    gains=utility(core.tmm(a[ii],b[ii],dd)[0])-baseline[ii];lookup={pair:i for i,pair in enumerate(pairs)}
    for i in range(len(a)):
     for m in range(2):
      if choice[i,m]>0:queried[i,m]=gains[lookup[(i,int(choice[i,m]))],m]
   accepted=np.where(queried>0,choice,0);bb=evaluate(truth,choice);vv=evaluate(truth,accepted)
   for kind,row in [('before',bb),('verified',vv)]:
    ref=expected['domains'][dom]['per_seed'][seed][kind]
    assert all(abs(row[k]-ref[k])<1e-10 for k in row),(dom,seed,kind,row,ref)
   assert vv['negative']==0 and vv['regret']<=bb['regret']+1e-12
   before.append(bb);after.append(vv)
  out[dom]={kind:{k:float(np.mean([r[k] for r in rows])) for k in rows[0]} for kind,rows in [('before',before),('verified',after)]}
 print(json.dumps(dict(status='PASS',new_training=False,scope='Archived frozen proposals, not a fresh test or CPU timing rerun',domains=out),indent=2))
if __name__=='__main__':main()
