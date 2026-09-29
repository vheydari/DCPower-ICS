"""Select whole archived fault events and all normal test rows.
Creates an index manifest, not a synthetic continuous timeline.
python low_prevalence.py --data-dir DATA --target .01 --seed 42 --output subset_1pct.csv
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def select(data_dir,target,seed):
 if not 0<target<1: raise ValueError('target must be between zero and one')
 d=pd.read_csv(data_dir/'dcpower_test.csv');meta=json.loads((data_dir/'dcpower_meta.json').read_text())
 normal=d.label.eq(0).to_numpy();n0=int(normal.sum());budget=int(np.floor(target*n0/(1-target)))
 events=meta['test_attack_log'];rng=np.random.default_rng(seed);reachable={0: []}
 for ei in rng.permutation(len(events)):
  e=events[int(ei)];mask=(d.timestamp>=e['start_s'])&(d.timestamp<e['start_s']+e['duration_s']);n=int(mask.sum())
  assert d.loc[mask,'label'].eq(1).all() and d.loc[mask,'attack_scenario'].eq(e['scenario']).all()
  # Subset-sum: maximize retained whole-event duration without exceeding budget.
  for total, prior in list(reachable.items()):
   if total+n<=budget and total+n not in reachable:
    reachable[total+n]=prior+[int(ei)]
 used=max(reachable);chosen=reachable[used]
 selected=normal.copy();event_id=np.full(len(d),-1)
 for i in chosen:
  e=events[i];mask=np.asarray((d.timestamp>=e['start_s'])&(d.timestamp<e['start_s']+e['duration_s']));selected|=mask;event_id[mask]=i
 idx=np.flatnonzero(selected);out=d.iloc[idx][['timestamp','label','attack_scenario']].copy();out.insert(0,'source_row',idx);out['event_id']=event_id[idx]
 # Gaps are segment boundaries. Never join across them in a temporal detector.
 out['block_id']=np.cumsum(np.r_[True,np.diff(idx)!=1])-1
 report={'target':target,'selection_seed':seed,'normal_rows':n0,'positive_budget':budget,'positive_rows':used,'actual_prevalence':used/len(out),'selected_event_indices':chosen,'scenario_coverage':int(out.loc[out.label==1,'attack_scenario'].nunique()),'independent_blocks':int(out.block_id.nunique())}
 assert used<=budget and len(out)==n0+used
 return out,report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--target',type=float,required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 out,report=select(a.data_dir,a.target,a.seed);a.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.output,index=False);a.output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
