const $ = (id) => document.getElementById(id);
let selectedFile = null;
let reports = [];
let appliedProfileContext = '';
let xaiBusy = false;
let xaiQueue = Promise.resolve();
let analysisController = null;
let xaiController = null;

const routeFor = (file) => {
  if (!file) return 'No file attached';
  const ext = file.name.split('.').pop().toLowerCase();
  if (['tif', 'tiff'].includes(ext)) return 'NeuroVision MRI';
  if (['png', 'jpg', 'jpeg', 'webp', 'bmp'].includes(ext)) return 'MedGemma vision';
  if (ext === 'pdf') return 'Qwen report text';
  if (['txt', 'md', 'csv'].includes(ext)) return 'Qwen text review';
  return 'Unsupported file';
};
const isMriFile = (file) => !!file && ['tif', 'tiff'].includes(file.name.split('.').pop().toLowerCase());
function toast(message) {
  const node = $('toast'); node.textContent = message; node.classList.add('show');
  clearTimeout(toast.timer); toast.timer = setTimeout(() => node.classList.remove('show'), 3600);
}
function userFacingError(error, context='review'){
  const message=String(error?.message||'');
  if(error?.name==='SyntaxError')return 'The local service returned an unreadable response. Refresh service checks and retry.';
  if(/failed to fetch|networkerror|connection refused|econnrefused/i.test(message)){
    return context==='xai'
      ? 'Could not reach NeuroVision on port 8080. Check the MRI service, then retry these visualizations.'
      : isMriFile(selectedFile)
        ? 'Could not reach NeuroVision on port 8080. Check the MRI service, then retry this slice.'
        : 'Could not reach the MedNexus service. Confirm the site is running, then retry.';
  }
  return message||'The local service could not complete this request. Check service status and retry.';
}
function formatBytes(n) { return n < 1024 * 1024 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`; }
function setFile(file) {
  selectedFile = file || null;
  const preview = $('filePreview');
  preview.classList.toggle('hidden', !file);
  if (file) {
    $('fileName').textContent = file.name;
    $('fileSize').textContent = formatBytes(file.size);
    $('fileType').textContent = file.name.split('.').pop().toUpperCase().slice(0, 5);
    $('fileRoute').textContent = routeFor(file);
  }
  renderFiles();
}
function acceptFile(file) {
  if (!file) return false;
  if (routeFor(file) === 'Unsupported file') {
    toast('This file type is not supported. Choose an MRI TIFF, image, PDF, TXT, MD, or CSV.');
    return false;
  }
  if (file.size > 12 * 1024 * 1024) {
    toast('This file is over the 12 MB limit. Choose a smaller file.');
    return false;
  }
  setFile(file);
  scrollToMessage($('composerWrap'));
  return true;
}
function showView(view) {
  document.querySelectorAll('.view').forEach(section => {
    const active = section.id === `view-${view}`;
    section.classList.toggle('active', active);
    section.hidden = !active;
  });
  document.querySelectorAll('.nav-item[data-view]').forEach(button => {
    const active = button.dataset.view === view;
    button.classList.toggle('active', active);
    if (active) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  });
  const titles = {chat:'Chats',profile:'Profile',reports:'Previous reports',files:'Your files',checks:'Service checks',safeguards:'Research safeguards'};
  $('pageTitle').textContent = titles[view] || 'Chats';
}
function appendMessage(kind, text, options = {}) {
  $('welcomeCard')?.remove();
  const article = document.createElement('article');
  article.className = `message ${kind}`;
  const label = document.createElement('p'); label.className = 'message-label';
  label.textContent = kind === 'user' ? 'YOU' : (options.label || 'MEDNEXUS · RESEARCH OUTPUT');
  const bubble = document.createElement('div'); bubble.className = 'message-bubble';
  if (options.error) bubble.classList.add('error');
  bubble.textContent = text;
  article.append(label, bubble);
  $('conversation').append(article);
  return article;
}
function addLoading() {
  const article = document.createElement('article'); article.className = 'message assistant'; article.id = 'pendingMessage';
  const label = document.createElement('p'); label.className = 'message-label'; label.textContent = isMriFile(selectedFile) ? 'NEUROVISION · MRI REVIEW' : 'MEDNEXUS · PREPARING REVIEW';
  const bubble = document.createElement('div'); bubble.className = 'message-bubble loading-line';
  const spinner = document.createElement('span'); spinner.className = 'spinner'; spinner.setAttribute('aria-hidden','true');
  const text = document.createElement('span'); text.textContent = isMriFile(selectedFile) ? 'Routing one 2D slice to the local MRI specialist…' : 'Preparing a local review path…';
  bubble.append(spinner, text); article.append(label, bubble); $('conversation').append(article); article.scrollIntoView({block:'nearest',behavior:'smooth'}); return article;
}
function scrollToMessage(node) {
  requestAnimationFrame(() => node.scrollIntoView({
    block: 'start',
    behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
  }));
}
function addFact(label, value, detail) {
  const card = document.createElement('div'); card.className = 'fact-card';
  const small = document.createElement('small'); small.textContent = label;
  const strong = document.createElement('b'); strong.textContent = value;
  const span = document.createElement('span'); span.textContent = detail;
  card.append(small, strong, span); return card;
}
function addTab(viewer, name, label, src, active = false) {
  const button = document.createElement('button'); button.type = 'button'; button.className = `view-tab${active ? ' active' : ''}`;
  button.dataset.imageName = name; button.dataset.imageSrc = src; button.textContent = label;
  button.setAttribute('aria-pressed', String(active)); viewer.toolbar.append(button); return button;
}
function setViewerImage(card, name, src, label, caption, alt) {
  const viewer = card.querySelector('.image-viewer');
  const img = viewer.querySelector('.scan-stage img');
  const empty = viewer.querySelector('.scan-empty');
  if (!src) {
    img.classList.add('hidden'); empty.classList.remove('hidden');
    empty.querySelector('b').textContent = 'No image was returned for this view';
    empty.querySelector('p').textContent = 'The specialist did not provide a preview for this output.';
  } else {
    img.src = src; img.alt = alt; img.classList.remove('hidden'); empty.classList.add('hidden');
  }
  viewer.querySelector('.viewer-caption strong').textContent = label;
  viewer.querySelector('.viewer-caption span').textContent = caption;
  viewer.toolbar.querySelectorAll('.view-tab').forEach(button => {
    const active = button.dataset.imageName === name;
    button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active));
  });
}
function renderXaiGallery(card, data) {
  const area = card.querySelector('.xai-area');
  const gallery = area.querySelector('.xai-gallery');
  const viewer = card.querySelector('.image-viewer');
  viewer.toolbar.querySelectorAll('.view-tab').forEach(tab => {
    if (tab.dataset.imageName !== 'segmentation') tab.remove();
  });
  const views = [
    {key:'original', title:'Original MRI', description:'The grayscale image submitted to NeuroVision.', meaning:'This is the source slice the model processed. It is here so you can compare the model overlays with the image itself.', limit:'A single image preview does not include the rest of the MRI sequences or the radiologist’s full-study interpretation.', caption:'Original MRI slice', alt:'Original brain MRI slice'},
    {key:'gradcam', title:'Where the classifier focused', description:'A heatmap linked to the classifier output.', meaning:'Warmer colors mark image areas more associated with the classifier’s tumour-category output in this run.', limit:'This is not a tumor outline or probability map, and it does not prove the highlighted tissue caused the result. The heatmap can be unstable.', caption:'Classifier Grad-CAM · model visualization', alt:'Grad-CAM heatmap over the brain MRI slice'},
    {key:'agreement', title:'How model outlines overlap', description:'Compares masks from the segmentation models.', meaning:'Green marks areas where all three segmentation models overlapped; yellow means two overlapped; red means one model marked the area.', limit:'Agreement only means the models drew similar outlines. They can agree and still be wrong; this is not a confidence score.', caption:'Segmentation agreement · model visualization', alt:'Segmentation ensemble agreement map'},
    {key:'uncertainty', title:'Where predictions varied', description:'Variation across repeated segmentation passes.', meaning:'Brighter areas show locations where repeated predictions from one segmentation model differed more for this slice.', limit:'This is not a validated uncertainty estimate and cannot tell how likely a diagnosis is.', caption:'Prediction variation · model visualization', alt:'MC-dropout variance visualization for the brain MRI slice'}
  ];
  gallery.replaceChildren();
  views.forEach(view => {
    const src = view.key === 'original' ? data.original : data.maps?.[view.key];
    const tile = document.createElement('article'); tile.className = 'xai-tile';
    const button = document.createElement('button'); button.type = 'button'; button.dataset.galleryView = view.key; button.setAttribute('aria-expanded','false'); button.setAttribute('aria-label', `Show ${view.title} and its explanation`);
    const image = document.createElement('img'); image.src = src || ''; image.alt = src ? view.alt : ''; image.width=256; image.height=256; image.loading='lazy';
    if (!src) image.classList.add('hidden');
    const title = document.createElement('h5'); title.textContent = view.title;
    const description = document.createElement('p'); description.textContent = src ? view.description : 'This view was not generated for this response.';
    button.append(image);
    if(!src){const unavailable=document.createElement('span');unavailable.className='xai-unavailable';unavailable.textContent='View unavailable';button.append(unavailable);}
    tile.append(button,title,description);
    const meaning = document.createElement('div'); meaning.className='xai-meaning hidden'; meaning.dataset.meaningFor=view.key;
    const closeMeaning=document.createElement('button');closeMeaning.type='button';closeMeaning.className='xai-meaning-close';closeMeaning.dataset.closeMeaning='true';closeMeaning.setAttribute('aria-label',`Close ${view.title} explanation`);closeMeaning.textContent='×';
    const meaningTitle=document.createElement('strong'); meaningTitle.textContent=src?'What this view shows':'View unavailable';
    const meaningText=document.createElement('p'); meaningText.textContent=src?view.meaning:'NeuroVision did not return this visualization for the current slice.';
    const limitTitle=document.createElement('strong'); limitTitle.textContent=src?'What it cannot tell you':'';
    const limitText=document.createElement('p'); limitText.textContent=src?view.limit:'';
    meaning.append(closeMeaning,meaningTitle,meaningText);if(src)meaning.append(limitTitle,limitText);tile.append(meaning);gallery.append(tile);
    if (src) addTab(viewer, view.key, ({original:'Original',gradcam:'Grad-CAM',agreement:'Agreement',uncertainty:'Variation'})[view.key], src);
  });
  const generatedCount = views.slice(1).filter(view => data.maps?.[view.key]).length;
  const retryButton = area.querySelector('.xai-button');
  retryButton.classList.toggle('hidden', generatedCount === 3);
  retryButton.disabled = false;
  retryButton.querySelector('span').textContent = generatedCount ? 'Regenerate XAI views' : 'Try generating the XAI views again';
  area.querySelector('.xai-heading p').textContent = generatedCount === 3
    ? 'Tap or click a view to see what it shows and what it cannot tell you.'
    : `${generatedCount} of 3 model views are available. Unreturned views are marked below.`;
  area.querySelector('.xai-foot').textContent = 'These pictures show how the model behaved on this slice. They are not a radiologist’s explanation or evidence of a diagnosis.';
  area.querySelector('.xai-disclaimer').textContent = 'The model views are generated automatically for this slice. They may be unstable; compare each one with the original image.';
  area.querySelector('.xai-gallery').classList.remove('hidden');
}
function buildMriCard(result, sourceFile) {
  const wrapper = document.createElement('article'); wrapper.className = 'review-card'; wrapper.dataset.mriCard = 'true';
  wrapper.mriFile = sourceFile;
  const head = document.createElement('header'); head.className = 'review-card-head';
  const titleWrap = document.createElement('div'); const title = document.createElement('h3'); title.textContent = 'MRI result · one uploaded image';
  const sub = document.createElement('p'); sub.textContent = 'This review used one image, not your complete MRI study.'; titleWrap.append(title,sub);
  const tag = document.createElement('span'); tag.className = 'source-tag mri-badge'; tag.textContent = 'Research output · not validated'; head.append(titleWrap,tag);
  const facts = document.createElement('div'); facts.className = 'review-facts';
  const predicted = result.predicted_class === 'tumour_suspected' ? 'Model flagged for review' : result.predicted_class === 'no_tumour_detected_by_model' ? 'No tumour detected by model in this slice' : (result.predicted_class || 'Not supplied');
  const score = typeof result.model_score === 'number' ? result.model_score.toFixed(3) : 'Not reported';
  const scoreMeaning = result.predicted_class === 'tumour_suspected'
    ? `Tumour-category score, not a ${score === 'Not reported' ? 'probability' : `${(Number(score) * 100).toFixed(1)}% chance`}.`
    : `Tumour-category score (${score}); not a rule-out test.`;
  const maskMeaning = result.segmentation_available
    ? 'Tint shows pixels the segmentation model marked as its target region; compare with the original.'
    : 'No shaded region was returned for this slice.';
  facts.append(addFact('MODEL RESULT',predicted,'A computer model’s label for this slice.'),addFact('MODEL SCORE',score,scoreMeaning),addFact('COLORED AREA',result.segmentation_available ? 'Model outline' : 'No outline returned',maskMeaning));
  const explanation = document.createElement('section'); explanation.className = 'mri-explanation';
  const explanationTitle = document.createElement('h4'); explanationTitle.textContent = 'In everyday language';
  const explanationBody = document.createElement('p');
  explanationBody.textContent = result.predicted_class === 'tumour_suspected'
    ? 'The model found a pattern in this single image that it has learned to associate with low-grade glioma (a type of brain tumor), so it marked the slice for review. That is a computer result, not confirmation that you have a tumor.'
    : result.predicted_class === 'no_tumour_detected_by_model'
      ? 'The model did not find enough of its learned pattern in this single image to flag it. That does not rule out a finding elsewhere in the MRI or replace the radiologist’s report.'
      : 'The model output is not available in a recognized form, so this interface cannot explain it reliably. Review the original scan and report with a qualified clinician.';
  const explanationDetails = document.createElement('p'); explanationDetails.className='mri-model-scope';
  explanationDetails.textContent = `This model was developed for low-grade glioma MRI slices. It sees one image at a time, not the complete scan, and its score is not a personal risk estimate. The full MRI and radiologist’s report are the right place to understand what was found.`;
  const clinicianQuestions = document.createElement('p'); clinicianQuestions.className = 'mri-clinician-questions';
  clinicianQuestions.textContent = 'A useful question for your clinician: “What does the report say about this area, and how does it fit with the rest of my MRI?”';
  explanation.append(explanationTitle, explanationBody, explanationDetails, clinicianQuestions);
  const viewer = document.createElement('section'); viewer.className = 'image-viewer'; viewer.setAttribute('aria-label','MRI image and model visualizations');
  const toolbar = document.createElement('div'); toolbar.className = 'viewer-toolbar'; viewer.toolbar=toolbar;
  const toolbarLabel = document.createElement('span'); toolbarLabel.className = 'viewer-toolbar-label'; toolbarLabel.textContent='IMAGE VIEWS'; toolbar.append(toolbarLabel);
  const segmentationSrc = result.segmentation_overlay || '';
  const originalSrc = result.original_slice_preview || '';
  if (originalSrc) addTab(viewer,'original','Original',originalSrc,true);
  addTab(viewer,'segmentation','Segmentation',segmentationSrc,!originalSrc && !!segmentationSrc);
  const stage = document.createElement('div'); stage.className='scan-stage';
  const initialSrc = originalSrc || segmentationSrc;
  const image = document.createElement('img'); image.className=initialSrc?'':'hidden'; image.src=initialSrc; image.alt=originalSrc?'Original brain MRI slice supplied to NeuroVision':'MRI slice with model-generated segmentation overlay'; image.width=result.image_width||256; image.height=result.image_height||256; image.loading='eager';
  const empty=document.createElement('div'); empty.className=`scan-empty${initialSrc?' hidden':''}`; const emptyTitle=document.createElement('b'); emptyTitle.textContent='No MRI preview returned'; const emptyText=document.createElement('p'); emptyText.textContent='The original slice is unavailable in this response. The model output still requires clinician review.'; empty.append(emptyTitle,emptyText); stage.append(image,empty);
  const caption=document.createElement('div'); caption.className='viewer-caption'; const captionStrong=document.createElement('strong'); captionStrong.textContent=originalSrc?'Original MRI slice':segmentationSrc?'Segmentation mask overlay':'Segmentation unavailable'; const captionSpan=document.createElement('span'); captionSpan.textContent=originalSrc?'Source slice supplied to the model':'Model-generated region visualization'; caption.append(captionStrong,captionSpan);
  viewer.append(toolbar,stage,caption);
  const xai=document.createElement('section'); xai.className='xai-area';
  const xaiHead=document.createElement('div'); xaiHead.className='xai-heading'; const xaiIntro=document.createElement('div'); const xaiTitle=document.createElement('h4'); xaiTitle.textContent='What the model highlighted'; const xaiSub=document.createElement('p'); xaiSub.textContent='Preparing additional views for this slice automatically.'; xaiIntro.append(xaiTitle,xaiSub);
  const xaiButton=document.createElement('button'); xaiButton.className='xai-button'; xaiButton.type='button'; xaiButton.dataset.action='generate-xai';
  xaiButton.innerHTML='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v3M12 18v3M3 12h3m12 0h3M5.6 5.6l2.1 2.1m8.6 8.6 2.1 2.1m0-12.8-2.1 2.1m-8.6 8.6-2.1 2.1"/><circle cx="12" cy="12" r="4"/></svg><span>Preparing views…</span>';
  xaiHead.append(xaiIntro,xaiButton);
  const disclaimer=document.createElement('p'); disclaimer.className='xai-disclaimer'; disclaimer.textContent='These views load automatically and may take a little longer. Your MRI result above is ready to read now.';
  const gallery=document.createElement('div'); gallery.className='xai-gallery hidden'; const foot=document.createElement('p'); foot.className='xai-foot';
  xai.append(xaiHead,disclaimer,gallery,foot);
  const scope=document.createElement('p'); scope.className='scope-note'; scope.textContent=result.scope || 'Research model; not validated for broad neurological diagnosis or patient-level diagnosis. This single slice cannot rule out findings in unexamined slices or modalities.';
  wrapper.append(head,facts,explanation,viewer,xai,scope);
  wrapper.addEventListener('click', async event => {
    const closeMeaning=event.target.closest('[data-close-meaning]');
    if(closeMeaning){const tile=closeMeaning.closest('.xai-tile');tile.classList.remove('has-open-meaning');tile.querySelector('.xai-meaning').classList.add('hidden');tile.querySelector('[data-gallery-view]').setAttribute('aria-expanded','false');return;}
    const tab=event.target.closest('.view-tab');
    if(tab){setViewerImage(wrapper,tab.dataset.imageName,tab.dataset.imageSrc,tab.textContent,tab.dataset.imageName==='original'?'Source slice supplied to the model':tab.dataset.imageName==='segmentation'?'Model-generated region visualization':'Model visualization; inspect against original',tab.textContent);return;}
    const tileButton=event.target.closest('[data-gallery-view]');
    if(tileButton){
      const tile=tileButton.closest('.xai-tile');const name=tileButton.dataset.galleryView;
      const details=tile.querySelector('.xai-meaning');const open=details.classList.contains('hidden');
      wrapper.querySelectorAll('.xai-tile.has-open-meaning').forEach(other=>{other.classList.remove('has-open-meaning');other.querySelector('.xai-meaning').classList.add('hidden');other.querySelector('[data-gallery-view]').setAttribute('aria-expanded','false');});
      details.classList.toggle('hidden',!open);tileButton.setAttribute('aria-expanded',String(open));
      tile.classList.toggle('has-open-meaning',open);
      const tabNode=Array.from(toolbar.querySelectorAll('.view-tab')).find(item=>item.dataset.imageName===name);
      if(tabNode)setViewerImage(wrapper,name,tabNode.dataset.imageSrc,tabNode.textContent,tile.querySelector('.xai-tile p').textContent,tabNode.textContent);
      return;
    }
    if(event.target.closest('[data-action="generate-xai"]')) await queueXai(wrapper);
    if(!event.target.closest('.xai-tile'))wrapper.querySelectorAll('.xai-tile.has-open-meaning').forEach(tile=>{tile.classList.remove('has-open-meaning');tile.querySelector('.xai-meaning').classList.add('hidden');tile.querySelector('[data-gallery-view]').setAttribute('aria-expanded','false');});
  });
  return wrapper;
}
function queueXai(card){
  xaiQueue=xaiQueue.then(()=>requestXai(card)).catch(error=>console.error('MRI visualization queue failed',error));
  return xaiQueue;
}
async function requestXai(card) {
  if(!card.isConnected)return;
  if(xaiBusy || !isMriFile(card.mriFile)){toast('This MRI source is no longer available in the current page session.');return;}
  xaiBusy=true;const button=card.querySelector('.xai-button');button.disabled=true;button.querySelector('span').textContent='Generating model visualizations…';
  card.querySelector('.xai-heading p').textContent='Preparing Grad-CAM, model-mask agreement, and prediction variation for this image. This can take a little while.';
  const controller=new AbortController();xaiController=controller;
  try{
    const body=new FormData();body.append('file',card.mriFile);
    const response=await fetch('/api/explain',{method:'POST',body,signal:controller.signal});const data=await response.json();
    if(!response.ok)throw new Error(data.detail||data.error||'Could not generate the model visualizations.');
    renderXaiGallery(card,data);
    scrollToMessage(card.querySelector('.xai-gallery'));
    const originalTab=Array.from(card.querySelectorAll('.view-tab')).find(item=>item.dataset.imageName==='original');
    if(originalTab)setViewerImage(card,'original',originalTab.dataset.imageSrc,'Original','Reference image used for this model pass.','Original brain MRI slice');
    const report=reports.slice().reverse().find(item=>item.file===card.mriFile);if(report)report.xai='generated';renderReports();
  }catch(error){
    if(error.name!=='AbortError'){card.querySelector('.xai-heading p').textContent=`The MRI result is still available. Visualizations could not be loaded: ${userFacingError(error,'xai')}`;button.querySelector('span').textContent='Retry visualizations';button.disabled=false;}
  }finally{if(xaiController===controller){xaiBusy=false;xaiController=null;}}
}
function renderFiles(){
  const panel=$('filesPanel');if(!panel)return;
  const files=[...new Set([...reports.map(report=>report.file),selectedFile].filter(Boolean))];
  if(!files.length){panel.className='surface-card';panel.innerHTML='<div class="empty-card"><span class="empty-kicker">NO FILES YET</span><b>Your session starts with one source</b><p>Add an MRI slice, image, or report from Chats. Attached files and review outputs remain only in this browser session.</p><button class="primary-action" type="button" data-view-jump="chat">Open Chats to add a file <span>→</span></button></div>';return;}
  panel.className='file-list';const heading=document.createElement('div');heading.className='list-summary';heading.innerHTML=`<strong>${files.length} ${files.length===1?'source':'sources'}</strong><span>Temporary to this browser session</span>`;panel.replaceChildren(heading,...files.map(file=>{const row=document.createElement('article');row.className='file-row';const icon=document.createElement('span');icon.className='file-row-icon';icon.textContent=file.name.split('.').pop().toUpperCase().slice(0,4);const main=document.createElement('div');main.className='file-row-main';const name=document.createElement('b');name.textContent=file.name;const meta=document.createElement('small');meta.textContent=`${formatBytes(file.size)} · ${routeFor(file)}`;main.append(name,meta);const status=document.createElement('span');status.className=`file-row-status${reports.some(report=>report.file===file)?' reviewed':''}`;status.textContent=reports.some(report=>report.file===file)?'Reviewed':'Ready';row.append(icon,main,status);return row;}));
}
function renderReports(){
  const panel=$('reportsPanel');if(!panel)return;
  const count=$('reportCount');if(count)count.textContent=reports.length?`${reports.length} ${reports.length===1?'review output':'review outputs'} · this session only`:'No review outputs yet';
  if(!reports.length){panel.innerHTML='<div class="empty-card"><span class="empty-kicker">NO REVIEWS YET</span><b>Your research outputs will appear here</b><p>Start with a question or add one source in Chats. Each result stays in this session and should be reviewed with the original material.</p><button class="primary-action" type="button" data-view-jump="chat">Start a review <span>→</span></button></div>';return;}
  panel.replaceChildren(...reports.slice().reverse().map(report=>{const article=document.createElement('article');article.className='report-entry';const head=document.createElement('div');head.className='report-entry-head';const title=document.createElement('h2');title.textContent=report.title;const time=document.createElement('time');time.textContent=report.time;head.append(title,time);const meta=document.createElement('div');meta.className='report-source';meta.textContent=[report.provenance,report.mode==='simulated_demo'?'Simulated example':report.mode==='live_model'?'Live local model':'Research output'].filter(Boolean).join(' · ');const source=document.createElement('div');source.className='report-source';source.textContent=report.file?`Source · ${report.file.name}`:'Source · question and session context';const body=document.createElement('p');body.textContent=report.summary;article.append(head,meta,source,body);if(report.scope){const scope=document.createElement('p');scope.className='report-scope';scope.textContent=`Scope · ${report.scope}`;article.append(scope);}if(report.title.startsWith('NeuroVision')){const note=document.createElement('p');note.className='report-xai';note.textContent=report.xai?'Exploratory NeuroVision views were generated for this slice; click each view in Chats for its plain-language explanation.':'Grad-CAM and related views load automatically after an MRI result; click each view in Chats to read its explanation.';article.append(note);}return article;}));
}
function renderResult(data, userText, sourceFile){
  const mri=data.specialist==='NeuroVision MRI';
  const resultText=data.summary||'The specialist returned no summary. Review the source material and check the configured services.';
  const specialist=data.specialist||'Configured local specialist';
  const result=data.specialist_result||{};
  const mriReply = result.predicted_class === 'tumour_suspected'
    ? 'NeuroVision flagged this slice for additional review. The explanation below separates the model output from what can be concluded clinically.'
    : result.predicted_class === 'no_tumour_detected_by_model'
      ? 'NeuroVision did not flag this slice. The explanation below covers why that does not rule out findings elsewhere in the study.'
      : 'NeuroVision returned a slice-level research result. The explanation below identifies what is known and what is not available.';
  const assistant=appendMessage('assistant',mri?mriReply:'The configured specialist returned a preliminary summary.',{label:data.provenance||`${specialist.toUpperCase()} · RESEARCH OUTPUT`});
  if(mri){const card=buildMriCard(result,sourceFile);assistant.append(card);if(sourceFile)queueXai(card);}
  if(!mri){const summaryCard=document.createElement('section');summaryCard.className='review-card';const summaryHead=document.createElement('header');summaryHead.className='review-card-head';const summaryTitle=document.createElement('h3');summaryTitle.textContent=`Review summary · ${specialist}`;const provenance=document.createElement('span');provenance.className='source-tag';provenance.textContent=data.mode==='simulated_demo'?'SIMULATED FIXTURE':'LOCAL MODEL OUTPUT';summaryHead.append(summaryTitle,provenance);const summary=document.createElement('div');summary.className='result-summary';summary.textContent=resultText;summaryCard.append(summaryHead,summary);assistant.append(summaryCard);}
  if(data.scope&&!mri){const scope=document.createElement('p');scope.className='scope-note';scope.textContent=data.scope;assistant.append(scope);}
  const mriReportSummary=mri
    ? `${mriReply} The model score (${typeof result.model_score==='number'?result.model_score.toFixed(3):'not reported'}) is not a personal chance or risk estimate. This model was developed for low-grade glioma MRI slices and reviewed only this one image; use the full MRI study and radiologist’s report for clinical meaning.`
    : resultText;
  const report={title:mri?'NeuroVision MRI · one image':`${data.specialist||'Model'} · review`,summary:mriReportSummary,time:new Date().toLocaleString([],{month:'short',day:'numeric',hour:'numeric',minute:'2-digit'}),xai:null,file:sourceFile,provenance:data.provenance||'RESEARCH OUTPUT',mode:data.mode||'research',scope:data.scope||''};reports.push(report);
  if(reports.length===1){const title=userText?.slice(0,36).trim()||sourceFile?.name||'Evidence review';$('sessionTitle').textContent=title.length===36?`${title}…`:title;}
  $('chatCount').textContent='1';renderFiles();renderReports();
}
function addFailureActions(message, attempt){
  const actions=document.createElement('div');actions.className='error-actions';
  const check=document.createElement('button');check.type='button';check.className='error-action';check.textContent='Check services';check.addEventListener('click',()=>{showView('checks');refreshHealth();});
  const retry=document.createElement('button');retry.type='button';retry.className='error-action primary';retry.textContent='Retry this review';retry.addEventListener('click',()=>analyze(null,attempt));
  actions.append(check,retry);message.append(actions);
}
async function analyze(event, retryAttempt=null){
  event?.preventDefault();
  $('composerForm').querySelector('.context-disclosure').open = false;
  const attempt=retryAttempt||{
    symptoms:$('messageInput').value.trim(),
    history:$('history').value.trim(),
    file:selectedFile,
  };
  if(!attempt.symptoms&&!attempt.history&&!attempt.file){toast('Add a question, context, or one file to start.');return;}
  if(attempt.file&&routeFor(attempt.file)==='Unsupported file'){toast('This file type is not supported. Choose an MRI TIFF, image, PDF, TXT, MD, or CSV.');return;}
  if(attempt.file&&attempt.file.size>12*1024*1024){toast('This file is over the 12 MB limit. Choose a smaller file.');return;}
  attempt.userText=attempt.userText||[attempt.symptoms,attempt.file?`Attached: ${attempt.file.name}`:''].filter(Boolean).join('\n');
  if(!retryAttempt&&attempt.userText)appendMessage('user',attempt.userText);
  $('analyzeBtn').disabled=true;const pending=addLoading();
  const body=new FormData();body.append('symptoms',attempt.symptoms);body.append('history',attempt.history);if(attempt.file)body.append('file',attempt.file);
  const controller=new AbortController();analysisController=controller;
  try{
    const response=await fetch('/api/analyze',{method:'POST',body,signal:controller.signal});
    const data=await response.json();
    if(!response.ok)throw new Error(data.detail||'Review request failed. Check the local service status, then retry.');
    pending.remove();renderResult(data,attempt.userText,attempt.file);
    scrollToMessage($('conversation').lastElementChild);
    if($('messageInput').value.trim()===attempt.symptoms)$('messageInput').value='';
    if($('history').value.trim()===attempt.history)$('history').value='';
  }catch(error){
    pending.remove();
    if(error.name!=='AbortError'){
      const failure=appendMessage('assistant',userFacingError(error),{error:true,label:'REVIEW UNAVAILABLE'});
      addFailureActions(failure,attempt);
    }
  }finally{
    if(analysisController===controller){analysisController=null;$('analyzeBtn').disabled=false;}
  }
}
async function refreshHealth(){
  try{const response=await fetch('/api/health');const health=await response.json();const ollama=health.ollama?.connected;const mri=!!health.neurovision?.health_route_detected;
    $('healthUpdated').textContent=`Last checked · ${new Date().toLocaleTimeString([],{hour:'numeric',minute:'2-digit'})}`;
    $('connectionDot').className=`status-dot ${ollama||mri?'green':'muted'}`;$('connectionText').textContent=ollama||mri?'Local service responding':'Local services offline';
    const models=health.ollama?.models||[];const expected=health.expected_models||{};const hasModel=(wanted)=>!!wanted&&models.some(name=>name===wanted||(!wanted.includes(':')&&name.startsWith(`${wanted}:`)));const hasText=hasModel(expected.text||'qwen3.5:0.8b');const hasVision=hasModel(expected.vision||'medgemma1.5');
    $('textState').textContent=hasText?'ready':ollama?'model missing':'offline';$('visionState').textContent=hasVision?'ready':ollama?'model missing':'offline';$('textNote').textContent=hasText?`Connected model: ${expected.text||'Qwen'}. Human review is required.`:ollama?`Ollama responds, but ${expected.text||'the configured text model'} is not listed.`:'Start Ollama and refresh checks.';$('visionNote').textContent=hasVision?`Connected model: ${expected.vision||'MedGemma'}. Outputs remain preliminary.`:ollama?`Ollama responds, but ${expected.vision||'the configured vision model'} is not listed.`:'Start Ollama and refresh checks.';
    $('mriState').textContent=mri?(health.neurovision?.device==='cuda'?'GPU ready':'responding'):'offline';$('mriNote').textContent=mri&&health.neurovision?.device==='cuda'?`Using ${health.neurovision.gpu_name||'CUDA GPU'}. Connectivity only; clinical validation is not established.`:(health.neurovision?.note||'Connectivity check only; MRI inference still requires the service.');
  }catch{$('connectionText').textContent='Backend unavailable';$('connectionDot').className='status-dot muted';$('textState').textContent='offline';$('visionState').textContent='offline';$('mriState').textContent='offline';$('textNote').textContent='Start the MedNexus server and refresh checks.';$('visionNote').textContent='Start the MedNexus server and refresh checks.';$('mriNote').textContent='Start the MedNexus server and refresh checks.';$('healthUpdated').textContent='Could not reach the backend · check the local server and retry';}
}
function newSession(){
  analysisController?.abort();analysisController=null;xaiController?.abort();xaiController=null;
  selectedFile=null;reports=[];xaiBusy=false;appliedProfileContext='';$('profileName').value='';updateProfileIdentity();$('profileAge').value='';$('profileOtherConditions').value='';document.querySelectorAll('input[name="profileCondition"]').forEach(input=>input.checked=false);$('analyzeBtn').disabled=false;setFile(null);$('fileInput').value='';$('messageInput').value='';$('history').value='';$('profileContext').value='';updateContextCount();$('sessionTitle').textContent='New conversation';$('chatCount').textContent='1';$('conversation').replaceChildren();
  const welcome=document.createElement('article');welcome.className='welcome-card';welcome.id='welcomeCard';welcome.innerHTML='<div class="welcome-orbit" aria-hidden="true"><span></span><i></i><i></i></div><div class="welcome-copy"><span class="welcome-kicker">START WITH WHAT YOU HAVE</span><h2>What would you like to understand?</h2><p>Share a question or add one report or image. MedNexus routes it to the configured local specialist and keeps the model’s source and limitations visible.</p><div class="welcome-chips"><button type="button" class="prompt-chip" data-prompt="Help me understand the wording in this report.">Understand a report <span>↗</span></button><button type="button" class="prompt-chip" data-prompt="What should I ask my clinician about this MRI slice?">Prepare questions <span>↗</span></button></div></div><div class="welcome-foot"><span class="status-dot green"></span> Your material is sent only to the configured local review path.</div>';
  $('conversation').append(welcome);showView('chat');renderFiles();renderReports();
}
document.querySelectorAll('.nav-item[data-view]').forEach(button=>button.addEventListener('click',()=>showView(button.dataset.view)));
$('composerForm').addEventListener('submit',analyze);
$('fileInput').addEventListener('change',event=>{if(!acceptFile(event.target.files[0]))event.target.value='';});
$('removeFile').addEventListener('click',()=>{setFile(null);$('fileInput').value='';});
$('newSession').addEventListener('click',newSession);
$('sessionLink').addEventListener('click',()=>showView('chat'));
$('refreshStatus').addEventListener('click',refreshHealth);$('checksRefresh').addEventListener('click',refreshHealth);
$('saveContext').addEventListener('click',()=>{if(!$('profileAge').checkValidity()){$('profileAge').reportValidity();return;}const profile=getProfileContext();const existing=$('history').value.trim();const manual=existing===appliedProfileContext?'':existing;$('history').value=[manual,profile].filter(Boolean).join('\n\n');appliedProfileContext=profile;showView('chat');toast(profile?'Age and health context added to this session. The name stays in this browser only.':'No age or health context added. The name stays in this browser only.');$('messageInput').focus();});
$('profileContext').addEventListener('input',updateContextCount);$('profileAge').addEventListener('input',updateContextCount);$('profileOtherConditions').addEventListener('input',updateContextCount);document.querySelectorAll('input[name="profileCondition"]').forEach(input=>input.addEventListener('change',updateContextCount));
$('profileName').addEventListener('input',updateProfileIdentity);
$('clearContext').addEventListener('click',()=>{if(appliedProfileContext){$('history').value=$('history').value.split('\n\n').filter(section=>section!==appliedProfileContext).join('\n\n');appliedProfileContext='';}$('profileName').value='';updateProfileIdentity();$('profileAge').value='';$('profileOtherConditions').value='';$('profileContext').value='';document.querySelectorAll('input[name="profileCondition"]').forEach(input=>input.checked=false);updateContextCount();toast('Review profile cleared.');$('profileName').focus();});
document.addEventListener('click',event=>{const prompt=event.target.closest('[data-prompt]');if(prompt){$('messageInput').value=prompt.dataset.prompt;showView('chat');$('messageInput').focus();}});
document.addEventListener('click',event=>{const jump=event.target.closest('[data-view-jump]');if(jump)showView(jump.dataset.viewJump);});
$('chatFrame').addEventListener('dragenter',event=>{
  if(!event.dataTransfer?.types?.includes('Files'))return;
  event.preventDefault();$('chatFrame').classList.add('is-dragging');$('dropOverlay').setAttribute('aria-hidden','false');
});
$('chatFrame').addEventListener('dragover',event=>{if(event.dataTransfer?.types?.includes('Files'))event.preventDefault();});
$('chatFrame').addEventListener('dragleave',event=>{
  if(event.relatedTarget&&$('chatFrame').contains(event.relatedTarget))return;
  $('chatFrame').classList.remove('is-dragging');$('dropOverlay').setAttribute('aria-hidden','true');
});
$('chatFrame').addEventListener('drop',event=>{
  if(!event.dataTransfer?.files?.length)return;
  event.preventDefault();$('chatFrame').classList.remove('is-dragging');$('dropOverlay').setAttribute('aria-hidden','true');
  if(event.dataTransfer.files.length>1){toast('Attach one file per review.');return;}
  acceptFile(event.dataTransfer.files[0]);
});
document.addEventListener('dragover',event=>{if(event.dataTransfer?.types?.includes('Files'))event.preventDefault();});
document.addEventListener('drop',event=>{if(event.dataTransfer?.types?.includes('Files')&&!$('chatFrame').contains(event.target)){event.preventDefault();toast('Drop the file inside the chat area to attach it.');}});
$('messageInput').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();$('composerForm').requestSubmit();}});
renderFiles();renderReports();refreshHealth();

function updateContextCount(){
  const count=$('contextCount');if(count){const text=getProfileContext();count.textContent=text?`${text.length} characters of context`:'No health context added';}
}
updateContextCount();

function getProfileContext(){
  const parts=[];
  const age=$('profileAge')?.value.trim();
  if(age)parts.push(`Age: ${age} years`);
  const conditions=Array.from(document.querySelectorAll('input[name="profileCondition"]:checked'),input=>input.value);
  if(conditions.length)parts.push(`Relevant health history: ${conditions.join(', ')}`);
  const other=$('profileOtherConditions')?.value.trim();
  if(other)parts.push(`Other relevant conditions or medication: ${other}`);
  const note=$('profileContext')?.value.trim();
  if(note)parts.push(`Additional context or question: ${note}`);
  return parts.join('\n');
}
function updateProfileIdentity(){
  const identity=$('profileIdentity');if(!identity)return;
  const name=$('profileName').value.trim();identity.textContent=name?`Current review profile · ${name} · name stays on this screen`:'';identity.classList.toggle('hidden',!name);
}
