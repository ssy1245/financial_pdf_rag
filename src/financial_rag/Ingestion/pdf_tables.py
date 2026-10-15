"""Conservative native, borderless financial tables anchored by explicit year columns.

Only standalone year headers and repeated aligned numeric rows qualify. Unmatched
lines stay in the original paragraph stream; no OCR or inferred financial values.
"""
import re
from .document import Cell, SourceLocation

YEAR = re.compile(r'(?:19|20)\d{2}')
NUMBER = re.compile(r'\(?[−–-]?\d[\d,]*(?:\.\d+)?\)?%?|[—–-]')


def year_tables(page, page_number, excluded_boxes):
    import pymupdf
    lines = []
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines', []):
            text = ''.join(s['text'] for s in line['spans']).strip()
            box = pymupdf.Rect(line['bbox'])
            if text and not any(b.intersects(box) for b in excluded_boxes):
                lines.append((box, text))
    # Sorting by raw y is unsafe: font/baseline jitter can reverse year columns.
    groups = []
    for box,text in sorted(lines, key=lambda x:(x[0].y0, x[0].x0)):
        center = (box.y0+box.y1)/2
        if groups and abs(center-groups[-1][0]) <= 2.5:
            groups[-1][1].append((box,text))
        else:
            groups.append((center, [(box,text)]))
    headers = []
    for i,(_,row) in enumerate(groups):
        years = sorted([(b,t) for b,t in row if YEAR.fullmatch(t)], key=lambda x:x[0].x0)
        if 2 <= len(years) <= 5 and len({t for _,t in years}) == len(years):
            # Genuine financial columns, not two nearby dates in running prose.
            if all(years[j+1][0].x0-years[j][0].x1 > 12 for j in range(len(years)-1)):
                headers.append((i, years))
    results = []
    for h,(start,years) in enumerate(headers):
        end = headers[h+1][0] if h+1<len(headers) else len(groups)
        rows=[]; consumed=[]; last_y=groups[start][0]
        left=years[0][0].x0
        header_items=groups[start][1]
        label_header=' '.join(t for b,t in sorted(header_items,key=lambda x:x[0].x0) if b.x1 < left-20)
        for y,items in groups[start+1:end]:
            if y-last_y > 45:
                break
            last_y=y
            values={}; labels=[]; used=[]; ambiguous=False
            for box,text in sorted(items,key=lambda x:x[0].x0):
                if box.x1 < left-20:
                    labels.append((box,text));continue
                if text in {'$', '€', '£'}:
                    continue
                if not NUMBER.fullmatch(text.replace(' ', '').lstrip('$€£')):
                    if text.strip(): ambiguous=True
                    continue
                nearest=min(range(len(years)), key=lambda j:abs(box.x1-years[j][0].x1))
                if abs(box.x1-years[nearest][0].x1)>18 or nearest in values:
                    ambiguous=True;continue
                values[nearest]=(box,text);used.append((box,text))
            label=' '.join(t for _,t in labels)
            if ambiguous or len(values)!=len(years) or not re.search(r'[A-Za-z]',label):
                continue
            currency_boxes=[]
            for currency_box,currency in items:
                if currency not in {'$', '€', '£'}:
                    continue
                matches=[j for j,(b,_) in values.items() if 0 <= b.x0-currency_box.x1 <= 24]
                if len(matches)==1:
                    j=matches[0];box,text=values[j]
                    values[j]=(box | currency_box, currency+' '+text)
                    currency_boxes.append(currency_box)
            rows.append((labels,values))
            consumed.extend(b for b,t in items if t not in {'$', '€', '£'} or b in currency_boxes)
        if len(rows)<3:
            continue
        consumed.extend(b for b,_ in header_items)
        bbox=[min(b.x0 for b in consumed),min(b.y0 for b in consumed),max(b.x1 for b in consumed),max(b.y1 for b in consumed)]
        source=SourceLocation(format='pdf',page=page_number,bbox=bbox)
        cells=[]
        def cell(r,c,text,boxes):
            box=[min(b.x0 for b in boxes),min(b.y0 for b in boxes),max(b.x1 for b in boxes),max(b.y1 for b in boxes)]
            cells.append(Cell(row=r,column=c,text=text,source=source.model_copy(update={
                'bbox':box,'row_start':r,'row_end':r,'column_start':c,'column_end':c})))
        label_boxes=[b for b,t in header_items if b.x1<left-20]
        cell(1,1,label_header,label_boxes or [pymupdf.Rect(bbox[0],bbox[1],left-20,years[0][0].y1)])
        for j,(box,year) in enumerate(years,2):cell(1,j,year,[box])
        for r,(labels,values) in enumerate(rows,2):
            cell(r,1,' '.join(t for _,t in labels),[b for b,_ in labels])
            for j,(box,text) in values.items():cell(r,j+2,text,[box])
        results.append(dict(rows=len(rows)+1,columns=len(years)+1,cells=cells,source=source,
                            header_rows=[1],header_origin='heuristic',
                            column_names=[label_header]+[t for _,t in years],consumed=consumed))
    return results
