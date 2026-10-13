"""Gold-aligned evidence-unit retrieval proxy, not semantic answer correctness.

Each unit must be supported inside returned contiguous source text on an annotated page of
its target report. Numeric units bind row labels and exact numbers locally.
Adjacent returned chunks may be joined without gaps; different units may be covered by different chunks. No page-only credit.
"""
import re
import unicodedata
from itertools import product


def phrase_pattern(value):
    # OCR/PDF spaces, hyphens and punctuation may split or join printed words.
    value=unicodedata.normalize('NFKC',value).casefold()
    # Preserve decimal points in factual phrases (13.2 must not match 132).
    chars=[re.escape(c) for i,c in enumerate(value) if c.isalnum() or
           (c=='.' and i>0 and i+1<len(value) and value[i-1].isdigit() and value[i+1].isdigit())]
    pattern=r'[\W_]*'.join(chars)
    significant=[c for c in value if c.isalnum()]
    if significant and significant[0].isdigit():pattern=r'(?<![\d.,])'+pattern
    if significant and significant[-1].isdigit():pattern+=r'(?!\d|[.,]\d)'
    return pattern


def number_pattern(value):
    value=str(value).replace(',','')
    parts=value.split('.')
    integer=parts[0]
    groups=[]
    while len(integer)>3:
        groups.insert(0,integer[-3:]);integer=integer[:-3]
    groups.insert(0,integer)
    body=r'[,\s]*'.join(map(re.escape,groups))
    if len(parts)>1:body+=r'\.'+re.escape(parts[1])
    return r'(?<![\d.,])'+body+r'(?!\d|[.,]\d)'


def match_unit(text, unit):
    """Return a minimal matching source span, or None; do not inject missing source text."""
    text=unicodedata.normalize('NFKC',text)
    groups=[]
    for anchor in unit['anchors']:
        pattern=anchor['regex'] if 'regex' in anchor else number_pattern(anchor['number']) if 'number' in anchor else '(?:'+'|'.join(phrase_pattern(p) for p in anchor['phrases'])+')'
        groups.append([(m.start(),m.end()) for m in re.finditer(pattern,text,re.I)])
    if any(not g for g in groups):return None
    # Window around each primary label; numbers must follow the label for rows.
    best=None
    for primary in groups[0]:
        chosen=[primary]
        for matches in groups[1:]:
            valid=[m for m in matches if max(primary[1],m[1])-min(primary[0],m[0])<=unit['max_span'] and
                   (not unit.get('values_after_label') or m[0]>=primary[1])]
            if not valid:break
            chosen.append(min(valid,key=lambda m:max(primary[1],m[1])-min(primary[0],m[0])))
        if len(chosen)!=len(groups):continue
        a=min(x[0] for x in chosen);b=max(x[1] for x in chosen)
        if b-a<=unit['max_span'] and (best is None or b-a<best[1]-best[0]):best=(a,b)
    return {'start':best[0],'end':best[1],'text':text[best[0]:best[1]]} if best else None


def retrieved_segments(chunks):
    """Join only returned source-adjacent spans on the same page/document."""
    groups={}
    for rank,c in enumerate(chunks,1):
        c=dict(c,text=unicodedata.normalize('NFKC',c['text']))
        groups.setdefault((c['document_id'],c['page']),[]).append((rank,c))
    segments=[]
    for (doc,page),group in groups.items():
        group.sort(key=lambda x:x[1].get('char_start',0))
        current=None
        for rank,c in group:
            adjacent=current is not None and 'char_start' in c and current.get('char_end')==c['char_start']
            part={'chunk_id':c['chunk_id'],'rank':rank,'start':len(current['text']) if adjacent else 0}
            part['end']=part['start']+len(c['text'])
            if adjacent:
                current['text']+=c['text'];current['char_end']=c['char_end'];current['parts'].append(part)
            else:
                current=dict(c,parts=[part]);segments.append(current)
    return segments


def score_content(chunks, criteria):
    if criteria.get('manual_required'):
        return {'hit':None,'coverage':None,'all_units':None,'matched_units':[],'missing_units':[], 'matches':[], 'manual_required':True}
    matches=[];covered=set()
    for chunk in retrieved_segments(chunks):
        if chunk['document_id']!=criteria['document_id']:continue
        for unit in criteria['units']:
            if chunk['page'] not in unit['pages']:continue
            span=match_unit(chunk['text'],unit)
            if span:
                parts=[p for p in chunk['parts'] if p['end']>span['start'] and p['start']<span['end']]
                covered.add(unit['id']);matches.append({'unit':unit['id'],'label':unit['label'],'rank':max(p['rank'] for p in parts),
                    'chunk_ids':[p['chunk_id'] for p in parts],'page':chunk['page'],'span':span})
    ids=[u['id'] for u in criteria['units']]
    return {'hit':int(bool(covered)),'coverage':len(covered)/len(ids),'all_units':int(len(covered)==len(ids)),
        'matched_units':sorted(covered),'missing_units':[u for u in ids if u not in covered],'matches':matches,'manual_required':False}
