"""Standalone quality replay of all fixed shortlists; requires only NumPy.

Does not retrain predictors or reproduce machine-specific timing.
"""
from pathlib import Path
import json
import numpy as np
import core
P=Path(__file__).resolve().parent
def utility(r):return np.stack((r[...,:15].mean(-1),r[...,26:].mean(-1)),-1)
def main():
 x=json.loads((P/'summary.json').read_text(encoding='utf8'))
 with np.load(P/'data/inputs_truth.npz') as z:a,b,d,y=[z[k] for k in ['a','b','d','truth']]
 arr=np.concatenate((np.zeros((len(y),1,2)),y.reshape(len(y),48,2)),1);best=arr.max(1);base=utility(core.tmm(a,b,d)[0]);maxerr=0.
 for key,ref in x['quality'].items():
  f=P/'data'/f'{key}_decision.npz'
  with np.load(f) as z:ranks=z['ranks'];original=z['choice'];pairs=z['pairs'];saved=z['gains']
  ii=pairs[:,0];kk=pairs[:,1];dd=d[ii].copy();dd[np.arange(len(ii)),(kk-1)//2]+=np.where((kk-1)%2==0,-.01,.01)
  gains=utility(core.tmm(a[ii],b[ii],dd)[0])-base[ii];maxerr=max(maxerr,float(abs(gains-saved).max()))
  v=np.zeros((len(a),ranks.shape[1]+1,2));lookup={tuple(pair):j for j,pair in enumerate(pairs)}
  for i in range(len(a)):
   for m in range(2):v[i,1:,m]=gains[[lookup[(i,int(k))] for k in ranks[i,:,m]],m]
  ids=np.concatenate((np.zeros((len(a),1,2),int),ranks),1);choice=np.take_along_axis(ids,v.argmax(1)[:,None,:],1)[:,0]
  assert np.array_equal(choice,original),key
  selected=np.take_along_axis(arr,choice[:,None,:],1)[:,0]
  metric=dict(gainrecovery=float(selected.sum()/best.sum()),regret=float((best-selected).mean()),negative=float((selected < -1e-10).mean()),besthit=float(((best-selected)<1e-10).mean()))
  assert all(abs(metric[k]-ref[k])<1e-10 for k in metric),(key,metric,ref)
 assert maxerr<1e-9
 print(json.dumps(dict(status='PASS',policies=len(x['quality']),objects=len(a),max_replayed_gain_error=maxerr,new_training=False,scope='Frozen archived shortlist decisions; quality replay only, no timing or model rerun'),indent=2))
if __name__=='__main__':main()
