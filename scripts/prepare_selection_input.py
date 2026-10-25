"""Join released candidate/features records into one public select_sql.py input."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True,help='Extracted reproducibility companion directory')
    p.add_argument('--pool',required=True)
    p.add_argument('--question-id',type=int,required=True)
    p.add_argument('--db-id')
    p.add_argument('--profile',default='full_lr',choices=['full_lr','a1_lite_lr','a2_full_minus_semantic_risk_lr','a3_execution_cluster_only_lr'])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    def rows(name):
        with (a.root/a.pool/name).open(encoding='utf-8') as f:
            return [r for line in f if line.strip() for r in [json.loads(line)]
                    if r['question_id']==a.question_id and (a.db_id is None or r['db_id']==a.db_id)]
    candidates=rows('candidates.jsonl');features=rows('features.jsonl')
    assert len(candidates)==5 and len({r['db_id'] for r in candidates})==1,'Specify one unique database/question with five slots'
    allowed=set(json.loads((a.root/'models'/(a.pool+'__'+a.profile+'.json')).read_text(encoding='utf-8'))['raw_feature_names'])
    by_id={r['candidate_id']:r['features'] for r in features}
    output=[]
    for r in sorted(candidates,key=lambda r:r['candidate_order']):
        output.append({'candidate_id':r['candidate_id'],'generation_mode':r['generation_mode'],'sql':r['sql'],
                       **{k:v for k,v in by_id[r['candidate_id']].items() if k[8:] in allowed}})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Wrote five candidate inputs without evaluation labels:',a.output)

if __name__=='__main__':main()
