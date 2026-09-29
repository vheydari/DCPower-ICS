"""Read-only release audit and maintenance feature ablation. No simulator changes.
Run: python revision_analysis.py --data-dir DATA --code-dir CODE --out-dir RESULTS
"""
import argparse, json, hashlib, platform, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

STATUS_COMMAND = ['utility_available','utility_degraded','PCC_breaker','PCC_breaker_cmd',
 'GEN1_cmd','GEN2_cmd','GEN1_running','GEN2_running','GEN1_ready','GEN2_ready',
 'GEN1_fault','GEN2_fault','GEN1_breaker','GEN2_breaker','UPS_bypass','BESS_cmd',
 'load_shed_stage','maintenance_bypass']

def masks(df, log):
 out={}
 for kind in ['generator_test','ups_bypass','load_shed_test']:
  m=np.zeros(len(df),bool)
  for e in log:
   if e['type']==kind: m |= (df.timestamp>=e['start_s']) & (df.timestamp<e['start_s']+e['duration_s'])
  out[kind]=np.asarray(m)
 out['maintenance']=out['generator_test']|out['ups_bypass']|out['load_shed_test']
 # Same explicit definition for every seed and feature set; flags define evaluation subsets only.
 out['grid'] = np.asarray((df.utility_available==1)&(df.utility_degraded==0)&
  (df.GEN1_cmd==0)&(df.GEN2_cmd==0)&(df.GEN1_P<5)&(df.GEN2_P<5)&
  (df.UPS_bypass==0)&(df.load_shed_stage==0)&(df.maintenance_bypass==0)&~out['maintenance'])
 return out

def scores(X, seed):
 models={
  'IForest':IsolationForest(n_estimators=200,contamination=.05,random_state=seed,n_jobs=-1),
  'PCA':PCA(n_components=10,svd_solver='full'),
  'AE':MLPRegressor(hidden_layer_sizes=(32,16,8,16,32),activation='relu',solver='adam',max_iter=50,
                   random_state=seed,early_stopping=True,validation_fraction=.1,n_iter_no_change=5)}
 for name,model in models.items():
  with warnings.catch_warnings(record=True) as ws:
   if name=='AE': model.fit(X,X)
   else: model.fit(X)
  s=-model.score_samples(X) if name=='IForest' else np.mean((X-(model.inverse_transform(model.transform(X)) if name=='PCA' else model.predict(X)))**2,axis=1)
  yield name,s,[str(w.message) for w in ws]

def audit(train,test,meta):
 out={}
 for split,d in [('train',train),('test',test)]:
  residual=d.PCC_P+d.GEN1_P+d.GEN2_P+d.BESS_P-d.UPS_in_P-d.P_cooling-d.P_noncritical
  out[split]={'rows':len(d),'columns':len(d.columns),'missing':int(d.isna().sum().sum()),
   'duplicate_timestamps':int(d.timestamp.duplicated().sum()),'dt':d.timestamp.diff().dropna().unique().tolist(),
   'positive_rows':int(d.label.sum()),'ranges':{c:[float(d[c].min()),float(d[c].max())] for c in meta['columns']['sensors']},
   'power_residual_kw':{'min':float(residual.min()),'max':float(residual.max()),'mean':float(residual.mean()),'negative_fraction':float((residual<0).mean())},
   'ups_ratio':{'min':float((d.UPS_out_P/d.UPS_in_P).min()),'max':float((d.UPS_out_P/d.UPS_in_P).max())},
   'generator_abs_step_max_kw':{c:float(d[c].diff().abs().max()) for c in ['GEN1_P','GEN2_P']}}
 out['metadata_keys']=list(meta)
 out['events']={k:{'n':sum(e['type']==k for e in meta['train_maintenance_log']),
  'rows':sum(e['duration_s'] for e in meta['train_maintenance_log'] if e['type']==k)} for k in ['generator_test','ups_bypass','load_shed_test']}
 out['fault_duration']=[min(e['duration_s'] for e in meta['test_attack_log']),max(e['duration_s'] for e in meta['test_attack_log'])]
 out['scenario_counts']=pd.Series([e['scenario'] for e in meta['test_attack_log']]).value_counts().to_dict()
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--code-dir',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(a.code_dir.resolve()));import generate_dataset as gen
 train=pd.read_csv(a.data_dir/'dcpower_train.csv');test=pd.read_csv(a.data_dir/'dcpower_test.csv');meta=json.loads((a.data_dir/'dcpower_meta.json').read_text())
 (a.out_dir/'audit.json').write_text(json.dumps(audit(train,test,meta),indent=2))
 allcols=meta['columns']['sensors'];continuous=[c for c in allcols if c not in STATUS_COMMAND]
 env={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,
 'excluded':STATUS_COMMAND,'measurements':continuous,'detector_seed':42,'data_seeds':[42,123,777],
 'data_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.data_dir.glob('*') if f.is_file()},
 'code_sha256':{f:hashlib.sha256((a.code_dir/f).read_bytes()).hexdigest() for f in ['plant_model.py','generate_dataset.py','scenario_library.py']}}
 (a.out_dir/'environment.json').write_text(json.dumps(env,indent=2))
 results=[]
 for ds in [42,123,777]:
  if ds==42: d=train;log=meta['train_maintenance_log']
  else:
   rows,log=gen.generate_train(24,1.,ds,verbose=False);d=pd.DataFrame(rows)
  mm=masks(d,log)
  for feat,cols in [('all40',allcols),('measurement22',continuous)]:
   X=StandardScaler().fit_transform(d[cols].to_numpy(dtype=np.float32))
   for model,s,ws in scores(X,42):
    # Fit uses the full normal split; thresholds use either grid or all normal rows.
    for calibration,cm in [('grid',mm['grid']),('all_normal',np.ones(len(d),bool))]:
     threshold=float(np.quantile(s[cm],.95));pred=s>threshold
     for kind,m in mm.items():
      results.append(dict(data_seed=ds,detector_seed=42,features=feat,detector=model,calibration=calibration,condition=kind,rows=int(m.sum()),flagged=int(pred[m].sum()),far_pct=float(100*pred[m].mean()),threshold=threshold,warnings='; '.join(ws)))
    pd.DataFrame(results).to_csv(a.out_dir/'maintenance_ablation.csv',index=False)
    print(ds,feat,model,[(k,round(100*(s[m]>np.quantile(s[mm['grid']],.95)).mean(),2)) for k,m in mm.items()],flush=True)
 print('DONE',flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
