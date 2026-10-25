"""Replay frozen LR selections with exported JSON parameters; Python stdlib only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

POOLS = ['bird_deepseek_v4_pro', 'bird_qwen36_flash', 'bird_qwen36_plus', 'spider']
PROFILES = ['full_lr', 'a1_lite_lr', 'a2_full_minus_semantic_risk_lr', 'a3_execution_cluster_only_lr']
EXPECTED = {'bird_deepseek_v4_pro': [1083, 963, 1003], 'bird_qwen36_flash': [1054, 950, 957],
            'bird_qwen36_plus': [1128, 1047, 1056], 'spider': [789, 732, 747]}
SENSITIVITY = {'original': [1083,963,1003,985,993,972,50,10],
              'remove_c5': [1058,955,995,981,989,978,54,14],
              'deduplicate': [1083,953,1001,989,1001,985,71,23]}

def read(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def key(r):
    return r['db_id'], r['question_id'], r['candidate_id']

def priority(mode):
    return {'query_plan':0,'decomposition':1,'direct_sample':2,'direct':3}.get(mode,4 if mode.startswith('semantic_contrast') else 9)

def profile_features(features, model):
    allowed = set(model['raw_feature_names'])
    return {k[8:]:v for k,v in features.items() if k.startswith('feature_') and k[8:] in allowed}

def score(features, model):
    values = profile_features(features, model)
    logit = 0.0
    # DictVectorizer CSR indices are sorted before sparse matrix multiplication.
    for name, scale, coefficient in zip(model['encoded_feature_names'],model['scale'],model['coefficients']):
        if '=' in name:
            raw, category = name.split('=',1)
            value = float(values.get(raw) == category)
        else:
            value = float(values.get(name,0))
        if value:
            logit += (value / scale) * coefficient
    logit += model['intercept']
    if logit >= 0:
        return 1 / (1 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1 + exponential)

def select(rows, model):
    scored=[]
    for row in rows:
        f = profile_features(row['features'], model)
        p = score(row['features'],model)
        sortkey=(p,int(f.get('cluster_size',0)),int(f.get('execution_ok',0)),
                 -float(f.get('risk_score',0)),-int(f.get('high_severity_failures',0)),
                 -priority(row['generation_mode']),-row['candidate_order'])
        scored.append((sortkey,row))
    return max(scored,key=lambda pair:pair[0])[1],{r['candidate_id']:k[0] for k,r in scored}

def majority(rows):
    signatures=[r['execution_group'] for r in rows if r['execution_status']=='success']
    if not signatures:return rows[0]
    winner=Counter(signatures).most_common(1)[0][0]
    return next(r for r in rows if r['execution_group']==winner)

def change_pool(rows, intervention):
    selected=[];seen=set()
    for row in rows:
        if intervention=='remove_c5' and row['candidate_id']=='c5':continue
        if intervention=='deduplicate' and row['sql'] in seen:continue
        seen.add(row['sql'])
        selected.append({**row,'features':dict(row['features'])})
    groups=Counter(r['execution_group'] for r in selected if r['execution_status']=='success' and r['execution_group'])
    maximum=max(groups.values(),default=0)
    for row in selected:
        group=row['execution_group'];size=groups.get(group,0) if row['execution_status']=='success' else 0
        row['features'].update(feature_cluster_size=size,feature_cluster_ratio=size/len(selected),
            feature_cluster_is_majority=int(size>0 and size==maximum),feature_cluster_max_size=maximum,
            feature_cluster_signature_seen=int(bool(group)))
    return selected

def load_pool(root,pool):
    d=root/pool
    candidates=read(d/'candidates.jsonl'); features={key(r):r['features'] for r in read(d/'features.jsonl')}
    groups={key(r):r for r in read(d/'execution_groups.jsonl')}
    evaluations={key(r):r for r in read(d/'evaluation.jsonl')}
    result=defaultdict(list)
    for row in candidates:
        k=key(row)
        result[k[:2]].append({**row,'features':features[k],
            'execution_group':groups[k]['execution_group'],'execution_status':groups[k]['execution_status'],
            'official_correct':evaluations[k]['official_correct'],'archived_score':evaluations[k]['full_lr_score']})
    for rows in result.values():rows.sort(key=lambda r:r['candidate_order'])
    return result

def replay(root, output=None):
    models={p.stem:json.loads(p.read_text(encoding='utf-8')) for p in (root/'models').glob('*.json')}
    report={'pools':{},'sensitivity':{},'models_retrained':False,'sql_executed':False}
    for pool in POOLS:
        q=load_pool(root,pool);model=models[pool+'__full_lr']
        archived={(r['db_id'],r['question_id']):r for r in read(root/pool/'selections.jsonl')}
        totals=Counter();maximum_error=0.0
        for k,rows in q.items():
            chosen,scores=select(rows,model);vote=majority(rows)
            assert chosen['candidate_id']==archived[k]['selectors']['full_lr']['candidate_id'],(pool,k,'selection')
            assert vote['candidate_id']==archived[k]['selectors']['majority_vote']['candidate_id'],(pool,k,'majority')
            maximum_error=max(maximum_error,max(abs(scores[r['candidate_id']]-r['archived_score']) for r in rows))
            totals.update(oracle=max(r['official_correct'] for r in rows),majority=vote['official_correct'],full_lr=chosen['official_correct'])
        assert maximum_error<1e-12,(pool,maximum_error)
        assert [totals[k] for k in ['oracle','majority','full_lr']]==EXPECTED[pool]
        report['pools'][pool]={'questions':len(q),**totals,'all_selected_slots_match':True,'maximum_score_error':maximum_error}
        if pool!='bird_deepseek_v4_pro':continue
        for intervention in SENSITIVITY:
            totals=Counter();records=[]
            for k,rows in q.items():
                changed=rows if intervention=='original' else change_pool(rows,intervention)
                vote=majority(changed);oracle=max(r['official_correct'] for r in changed)
                record={'db_id':k[0],'question_id':k[1],'intervention':intervention,
                        'retained_candidates':[r['candidate_id'] for r in changed],
                        'oracle_correct':oracle,'majority':{'candidate_id':vote['candidate_id'],'correct':vote['official_correct']},'selectors':{}}
                totals.update(oracle=oracle,majority=vote['official_correct'])
                for profile in PROFILES:
                    chosen,scores=select(changed,models[pool+'__'+profile]);correct=chosen['official_correct']
                    totals[profile]+=correct
                    record['selectors'][profile]={'candidate_id':chosen['candidate_id'],'correct':correct,'scores':scores}
                    if intervention=='original':
                        assert chosen['candidate_id']==archived[k]['selectors'][profile]['candidate_id'],(k,profile)
                    if profile=='full_lr':totals.update(recovered=int(correct and not vote['official_correct']),regressed=int(vote['official_correct'] and not correct))
                records.append(record)
            actual=[totals[k] for k in ['oracle','majority',*PROFILES,'recovered','regressed']]
            assert actual==SENSITIVITY[intervention],(intervention,actual,SENSITIVITY[intervention])
            report['sensitivity'][intervention]=dict(totals)
            if output:
                output.mkdir(parents=True,exist_ok=True)
                (output/(intervention+'.jsonl')).write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in records),encoding='utf-8')
    if output:(output/'replay_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output',type=Path,help='Optional destination for per-question sensitivity records and summary')
    a=p.parse_args();print(json.dumps(replay(a.root,a.output),indent=2))
