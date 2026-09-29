const $=id=>document.getElementById(id);let selected=[],ready=false,busy=false;
function status(text,error=false){$('status').textContent=text;$('status').classList.toggle('error',error)}
function controls(){ $('upload').disabled=busy||!selected.length;$('files').disabled=busy;$('clear').disabled=busy;$('question').disabled=busy||!ready;$('send').disabled=busy||!ready; }
async function api(path,options={}){const response=await fetch(path,options);let data;try{data=await response.json()}catch{throw Error('服务返回异常，请确认后端仍在运行')}if(!response.ok)throw Error(data.error||'请求失败');return data}
async function streamAnswer(question,onEvent){
  const response=await fetch('/api/ask/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question})});
  if(!response.ok){const data=await response.json();throw Error(data.error||'请求失败')}
  const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='',result=null;
  function consume(line){
    if(!line.trim())return;const event=JSON.parse(line);
    if(event.type==='error')throw Error(event.message);
    if(event.type==='done')result=event.result;else onEvent(event);
  }
  try{
    while(!result){const {value,done}=await reader.read();buffer+=decoder.decode(value,{stream:!done});
      let newline;while((newline=buffer.indexOf('\n'))>=0){consume(buffer.slice(0,newline));buffer=buffer.slice(newline+1)}
      if(done){if(buffer.trim())consume(buffer);break}
    }
    if(!result)throw Error('连接中断，回答尚未完成');return result;
  }finally{await reader.cancel();reader.releaseLock()}
}
function showDocuments(documents){ready=documents.length>0;$('documents').replaceChildren();for(const d of documents){const li=document.createElement('li');li.textContent=`▤ ${d.name} · ${d.pages} 页${d.empty_pages?' · '+d.empty_pages+' 页无文字':''}`;$('documents').append(li)}if(!ready){const li=document.createElement('li');li.textContent='还没有文档';$('documents').append(li)}controls()}
function renderPending(){
  $('pending').replaceChildren();
  for(const file of selected){
    const li=document.createElement('li'),name=document.createElement('span'),remove=document.createElement('button');
    name.textContent=file.name;
    remove.type='button';remove.className='text-button';remove.textContent='移除';
    remove.setAttribute('aria-label','移除 '+file.name);remove.disabled=busy;
    remove.onclick=()=>{if(busy)return;selected=selected.filter(item=>item!==file);renderPending();selectionStatus();controls()};
    li.append(name,remove);$('pending').append(li);
  }
}
function selectionStatus(){status(selected.length
  ?'待上传 '+selected.length+' 份文件，可继续添加。点击开始阅读，将追加到当前资料库，保留已阅读文档。'
  :'尚未选择文件，请添加 PDF。')}
function select(files){
  if(busy)return;
  const incoming=[...files];if(!incoming.length)return;
  // 同一次待上传队列中避免重复选择同一个文件；后端仍按内容哈希去重。
  const key=f=>JSON.stringify([f.name,f.size,f.lastModified]);
  const known=new Set(selected.map(key)),list=[...selected];
  for(const file of incoming){if(!known.has(key(file))){known.add(key(file));list.push(file)}}
  if(list.length>10||list.some(f=>!f.name.toLowerCase().endsWith('.pdf')||f.size===0||f.size>20*1024*1024)||list.reduce((sum,f)=>sum+f.size,0)>100*1024*1024){
    status('请选择最多 10 份非空 PDF，每份不超过 20 MB，合计不超过 100 MB；原待上传列表已保留。',true);return;
  }
  selected=list;renderPending();selectionStatus();controls();
}
$('files').onchange=e=>{select(e.target.files);e.target.value=''};$('drop').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('files').click()}};
for(const event of ['dragover','drop'])$('drop').addEventListener(event,e=>e.preventDefault());$('drop').ondragover=()=>$('drop').classList.add('drag');$('drop').ondragleave=()=>$('drop').classList.remove('drag');$('drop').ondrop=e=>{$('drop').classList.remove('drag');select(e.dataTransfer.files)};
$('upload').onclick=async()=>{busy=true;controls();renderPending();status('正在读取文档并建立索引，首次从本地加载模型可能需要一些时间…');const form=new FormData();selected.forEach(f=>form.append('files',f));try{const data=await api('/api/upload',{method:'POST',body:form});showDocuments(data.documents);selected=[];$('pending').replaceChildren();$('files').value='';status('已准备好，可以针对这 '+data.documents.length+' 份 PDF 提问。')}catch(e){status(e.message+'；已有资料库保持不变。',true)}finally{busy=false;renderPending();controls()}};
function message(role,text){const box=document.createElement('article');box.className='message '+role;const title=document.createElement('h3');title.textContent=role==='user'?'你':'阅财 · AI';const body=document.createElement('div');body.textContent=text;box.append(title,body);$('chat').append(box);return box}
$('form').onsubmit=async e=>{
  e.preventDefault();const question=$('question').value.trim();if(!question||busy||!ready)return;
  const started=performance.now();
  const elapsed=()=>((performance.now()-started)/1000).toFixed(1);
  busy=true;controls();message('user',question);
  let phase='请求已发送，正在等待处理',box=null,answerBody=null;
  const tick=()=>status(phase+'… 已用 '+elapsed()+' 秒');
  tick();const timer=setInterval(tick,100);
  try{
    const data=await streamAnswer(question,event=>{
      if(event.type==='progress'){phase=event.message;tick()}
      if(event.type==='answer_delta'){
        if(!box){box=message('assistant','');answerBody=box.children[1]}
        answerBody.textContent+=event.text;phase='正在生成回答';tick();
      }
      if(event.type==='answer_reset'&&answerBody)answerBody.textContent='';
    });
    if(!box){box=message('assistant','');answerBody=box.children[1]}
    answerBody.textContent=data.answer;
    for(const source of data.sources){
      const details=document.createElement('details'),summary=document.createElement('summary'),text=document.createElement('p');
      summary.textContent=`[${source.label}] ${source.filename} · PDF 第 ${source.page} 页`;
      text.textContent=source.text;details.append(summary,text);box.append(details);
    }
    if(data.citation_status!=='labels_valid'){
      const warning=document.createElement('p');warning.className='warning';warning.textContent='引用缺失或编号异常，请核对原文。';box.append(warning);
    }
    if(data.status!=='completed'){
      const notice=document.createElement('p');notice.className='hint';
      const reasons={
        search_limit:'已达到本次检索次数上限，以下回答基于已获取的证据。',
        step_limit:'已达到本次模型调用轮数上限，以下回答基于已获取的证据。',
        no_new_evidence:'本轮没有找到新增证据，以下回答基于已有证据。',
        evidence_limit:'已达到本次证据容量上限，未继续加入新片段。',
        message_limit:'上下文已达到容量上限，未能完成本次回答。',
        retrieval_failed:'检索执行失败，请重试；这不代表文档中没有答案。'
      };
      notice.textContent=reasons[data.stop_reason]||'本次检索受到限制，请结合回答中的证据说明阅读。';
      if(['message_limit','retrieval_failed'].includes(data.stop_reason))notice.className='warning';
      box.append(notice);
    }
    if(data.query_plan?.warning){const warning=document.createElement('p');warning.className='warning';warning.textContent=data.query_plan.warning;box.append(warning)}
    const trace=document.createElement('details'),summary=document.createElement('summary'),list=document.createElement('ol');
    summary.textContent='查看检索过程';
    for(const round of data.search_history||[]){
      const li=document.createElement('li'),roundDetails=document.createElement('details'),roundTitle=document.createElement('summary');
      roundTitle.textContent=round.query+' · 新增 '+round.new_count+' 条证据';roundDetails.append(roundTitle);
      for(const item of round.evidence||[]){
        const card=document.createElement('div');card.className='evidence-card';
        const heading=document.createElement('p');heading.className='evidence-heading';
        heading.textContent=`第 ${item.rank} 名 · [${item.citation_id}] ${item.filename} · PDF 第 ${item.page} 页 · ${item.used_in_answer?'回答已引用':'回答未引用'}`;
        const preview=document.createElement('p');preview.className='evidence-preview';
        preview.textContent=item.text.slice(0,180)+(item.text.length>180?'…':'');
        const full=document.createElement('details'),title=document.createElement('summary'),text=document.createElement('p');
        title.textContent='展开片段全文';text.textContent=item.text;full.append(title,text);
        card.append(heading,preview,full);roundDetails.append(card);
      }
      if(!round.new_count){const empty=document.createElement('p');empty.textContent='本轮没有新增片段。';roundDetails.append(empty)}
      li.append(roundDetails);list.append(li);
    }
    trace.append(summary,list);box.append(trace);
    $('question').value='';
    clearInterval(timer);const seconds=elapsed();
    const timing=document.createElement('p');timing.className='hint';
    timing.textContent='总用时 '+seconds+' 秒 · 模型调用 '+data.model_calls+' 次 · 检索 '+(data.search_attempts??(data.search_history||[]).length)+' 次';if(Number.isInteger(data.planning_calls)&&Number.isInteger(data.model_steps)){
      timing.textContent+='（查询规划 '+data.planning_calls+' 次，回答／工具决策 '+data.model_steps+' 次）';
    }
    box.append(timing);
    status((data.status==='completed'?'回答完成':'本次检索已停止')+' · 总用时 '+seconds+' 秒。点击引用可查看原文。');
    box.scrollIntoView({behavior:'smooth',block:'center'});
  }catch(e){clearInterval(timer);if(box){const warning=document.createElement('p');warning.className='warning';warning.textContent='回答未完成，以上内容仅为部分输出。';box.append(warning)}status(e.message+' · 已用 '+elapsed()+' 秒',true)}
  finally{clearInterval(timer);busy=false;controls()}
};
$('clear').onclick=async()=>{busy=true;controls();try{await api('/api/documents',{method:'DELETE'});showDocuments([]);$('chat').replaceChildren();status('资料库已清空。')}catch(e){status(e.message,true)}finally{busy=false;controls()}};
api('/api/status').then(data=>{showDocuments(data.documents);if(!data.key_ready)status('请在项目 .env 配置 DEEPSEEK_API_KEY，然后重启服务。',true);else if(ready)status('资料库已就绪，可以继续提问。')}).catch(e=>status(e.message,true));
