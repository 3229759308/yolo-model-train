const $ = id => document.getElementById(id);
let target = 'input', currentFolder = '', parentFolder = null, busy = false, loadingStatus = false;
async function api(url, data) {
  const response = await fetch(url, data === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || '请求失败');
  return result;
}
function renderVideos(videos) {
  $('video-list').replaceChildren();
  if (!videos.length) { $('video-list').textContent = '当前文件夹没有支持的视频'; return; }
  const title = document.createElement('div'); title.textContent = `发现 ${videos.length} 个视频`; $('video-list').append(title);
  videos.forEach(video => {
    const row = document.createElement('div'); row.className = 'video';
    const name = document.createElement('span'); name.textContent = video.name;
    const size = document.createElement('span'); size.textContent = (video.bytes / 1048576).toFixed(1) + ' MB';
    row.append(name,size); $('video-list').append(row);
  });
}
async function scanInput() {
  try { renderVideos((await api('/api/folders?path='+encodeURIComponent($('input').value))).videos); }
  catch(error) { $('video-list').textContent = error.message; }
}
async function browse(path) {
  $('picker-error').textContent = '';
  try {
    const info = await api('/api/folders?path='+encodeURIComponent(path));
    currentFolder=info.path; parentFolder=info.parent; $('folder-path').value=info.path;
    $('up').disabled=!info.parent; $('folder-count').textContent=`${info.folders.length} 个子文件夹 · ${info.videos.length} 个视频`;
    $('folders').replaceChildren();
    info.folders.forEach(name => {
      const button=document.createElement('button'); button.type='button'; button.className='folder'; button.textContent='▸  '+name;
      button.onclick=()=>browse(currentFolder.replace(/\/$/,'')+'/'+name); $('folders').append(button);
    });
    if(!info.folders.length) $('folders').textContent='没有子文件夹，可选择当前文件夹。';
  } catch(error) { $('picker-error').textContent=error.message; }
}
document.querySelectorAll('[data-browse]').forEach(button=>button.onclick=async()=>{
  target=button.dataset.browse; $('picker-title').textContent=target==='input'?'选择视频文件夹':'选择输出文件夹';
  $('select-folder').disabled=true; $('picker').showModal();
  const path=$(target).value;
  try { await api('/api/folders?path='+encodeURIComponent(path)); await browse(path); }
  catch { const config=await api('/api/config'); await browse(config.root); }
  $('select-folder').disabled=false;
});
$('close-picker').onclick=()=>$('picker').close();
$('up').onclick=()=>parentFolder&&browse(parentFolder);
$('go').onclick=()=>browse($('folder-path').value);
$('folder-path').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();browse(e.target.value);}};
$('select-folder').onclick=()=>{ $(target).value=currentFolder; $('picker').close(); if(target==='input') scanInput(); };
$('input').onchange=scanInput;
$('step').oninput=()=>{
  const n=Number($('step').value);
  $('step-hint').textContent=Number.isInteger(n)&&n>0?`抽取第 0、${n}、${n*2}、${n*3}… 帧；保留原始分辨率。`:'请输入大于 0 的整数。';
};
function controls(running){
  busy=running; $('start').disabled=running; $('stop').disabled=!running;
  ['input','output','step'].forEach(id=>$(id).disabled=running);
  document.querySelectorAll('[data-browse]').forEach(b=>b.disabled=running);
}
$('form').onsubmit=async event=>{
  event.preventDefault(); $('error').textContent=''; controls(true);
  try { await api('/api/start',{input:$('input').value,output:$('output').value,step:Number($('step').value)}); await poll(); }
  catch(error){ $('error').textContent=error.message; controls(false); }
};
$('stop').onclick=async()=>{
  try{ $('stop').disabled=true; await api('/api/stop',{}); }
  catch(error){$('error').textContent=error.message;}
};
async function poll(){
  if(loadingStatus) return;
  loadingStatus=true;
  try{
    const s=await api('/api/status'); controls(s.status==='running');
    $('badge').textContent=({idle:'等待开始',running:'正在抽帧',done:'已完成',cancelled:'已停止',error:'处理失败'})[s.status]||s.status;
    $('saved').textContent=s.saved.toLocaleString(); $('count').textContent=`${s.completed} / ${s.total}`;
    $('progress').value=s.progress; $('percent').textContent=s.progress+'%';
    $('current').textContent=s.current||'准备好后，开始第一次抽帧';
    const text=s.logs.length?s.logs.join('\n'):'任务运行后，这里会显示处理进度。';
    if($('logs').textContent!==text){$('logs').textContent=text;$('logs').scrollTop=$('logs').scrollHeight;}
    $('result').textContent=s.output?'输出位置：'+s.output:'';
  }catch(error){$('badge').textContent='连接已断开'; $('error').textContent='无法连接本地服务，请确认 app.py 仍在运行。';}
  finally{loadingStatus=false;}
}
(async()=>{
  try{const c=await api('/api/config');$('input').value=c.input;$('output').value=c.output;await scanInput();await poll();}
  catch(error){$('error').textContent=error.message;}
})();
setInterval(poll,800);
