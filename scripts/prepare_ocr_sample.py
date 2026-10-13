from pathlib import Path
import requests,json,random,re,hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
out=Path(__file__).resolve().parents[1]/'outputs/ocr_ablation/omnidoc100_20261013'
rev=json.loads((out/'dataset_revision.json').read_text())['revision']
data=json.loads((out/'ground_truth_full.json').read_text());rng=random.Random(20261013);rng.shuffle(data)
def group(p):
 name=Path(p['page_info']['image_path']).stem
 return re.sub(r'(?:_page)?_\d+$','',name)
def attr(p):return p['page_info']['page_attribute']
def tablecomplex(p):
 issues=set(attr(p).get('special_issue',[]))
 return bool(issues & {'table_span','table_wireless_line','table_fewer_line','table_omission_line'})
# Choose one page per inferred source document, avoiding dev/test document leakage.
quotas=[('research_report',40,lambda p:attr(p).get('data_source')=='research_report'),('complex_table',20,tablecomplex),('scan',20,lambda p:'fuzzy_scan' in attr(p).get('special_issue',[])),('other_layout',20,lambda p:True)]
selected=[];seen=set();stats={}
for label,n,predicate in quotas:
 candidates=[p for p in data if predicate(p) and group(p) not in seen]
 chosen=[]
 # Alternate English and Chinese where available, then fill remaining languages.
 for language in ['english','simplified_chinese']:
  for p in candidates:
   if len([x for x in chosen if attr(x).get('language')==language])>=n//2:break
   if group(p) not in seen and attr(p).get('language')==language:
    chosen.append(p);seen.add(group(p))
 for p in candidates:
  if len(chosen)>=n:break
  if group(p) not in seen:chosen.append(p);seen.add(group(p))
 selected.extend((label,p) for p in chosen);stats[label]=len(chosen)
for p in data:
 if len(selected)>=100:break
 if group(p) not in seen:selected.append(('quota_fill',p));seen.add(group(p))
assert len(selected)==100
(out/'images').mkdir(exist_ok=True)
manifest=[]
for i,(label,p) in enumerate(selected):
 rawname=Path(p['page_info']['image_path']).name
 item=dict(id=f'p{i+1:03d}',image=rawname,source_group=group(p),stratum=label,split='dev' if i%5==0 else 'test',attributes=attr(p))
 manifest.append(item)
(out/'ground_truth.json').write_text(json.dumps([p for _,p in selected],ensure_ascii=False,indent=2))
def download(item):
 path=out/'images'/item['image']
 if not path.exists():
  url=f'https://huggingface.co/datasets/opendatalab/OmniDocBench/resolve/{rev}/images/{requests.utils.quote(item["image"],safe="")}'
  r=requests.get(url,timeout=180);r.raise_for_status();path.write_bytes(r.content)
 item['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
 return item
with ThreadPoolExecutor(max_workers=6) as ex:
 for i,item in enumerate(ex.map(download,manifest),1):
  if i%10==0:print('images',i,flush=True)
(out/'sample_manifest.json').write_text(json.dumps(dict(seed=20261013,revision=rev,requested_quotas={x[0]:x[1] for x in quotas},actual_strata=dict(Counter(x['stratum'] for x in manifest)),document_group_rule='filename stem minus terminal _page_N or _N; one page per inferred group',pages=manifest),ensure_ascii=False,indent=2))
print('COMPLETE',Counter(x['stratum'] for x in manifest),Counter(x['attributes'].get('language') for x in manifest),flush=True)
