'use strict';
const maintenance={timer:null,pending:null,offline:false};
try{maintenance.pending=JSON.parse(sessionStorage.getItem('gpu-lab-maintenance')||'null')}catch(error){/* Storage may be unavailable. */}

function saveMaintenance(value){
  maintenance.pending=value;
  try{if(value)sessionStorage.setItem('gpu-lab-maintenance',JSON.stringify(value));else sessionStorage.removeItem('gpu-lab-maintenance')}catch(error){/* Reconnection still works for the open tab. */}
}
function maintenanceSchedule(){
  clearTimeout(maintenance.timer);
  maintenance.timer=setTimeout(pollMaintenance,2000);
}
async function pollMaintenance(){
  if(!maintenance.pending)return;
  try{
    const response=await fetch('/api/health',{cache:'no-store',signal:AbortSignal.timeout(5000)});
    if(!response.ok)throw new Error('Waiting for server');
    const health=await response.json();
    if(health.instance!==maintenance.pending.instance){
      saveMaintenance(null);
      location.reload();
      return;
    }
    const status=await api('/settings');
    maintenance.offline=false;
    if(location.hash==='#settings')renderSettings(status);
    if(!status.busy){saveMaintenance(null);return}
    if(maintenance.pending.id&&status.operation.id!==maintenance.pending.id){saveMaintenance(null);return}
  }catch(error){
    maintenance.offline=true;
    if(location.hash==='#settings')renderMaintenanceOffline();
  }
  maintenanceSchedule();
}
function renderMaintenanceOffline(){
  const elapsed=Math.round((Date.now()-(maintenance.pending?.started||Date.now()))/1000);
  $('#app').innerHTML=head('SETTINGS','Updating your workspace.','GPU Lab is temporarily unavailable while background maintenance continues.')+
    `<section class="panel"><div class="panel-body" role="status" aria-live="polite"><h2>Waiting for the server…</h2><p>The page will reconnect and reload automatically. Installation may take several minutes.</p><p class="inline-note">Waiting ${elapsed} seconds. You can leave this tab open.</p>${elapsed>180?notice('Taking longer than expected','If the server does not return, inspect ~/.local/share/gpu-lab/maintenance/ on the server and run journalctl --user -u gpu-lab. A failed installation may need terminal recovery.','warn'):''}</div></section>`;
}
function renderSettings(data){
  const op=data.operation,short=value=>value?value.slice(0,12):'Not recorded';
  const disabled=data.busy||!data.supported?'disabled':'';
  const checkText=data.update_available?'New code is available.':op.checked_at?'Your installed code matches the last update check.':'Check GitHub to see whether newer code is available.';
  $('#app').innerHTML=head('SETTINGS','Keep your lab up to date.','Check GitHub, update the application, and restart the service from this browser.')+
    `${!data.supported?notice('One-time server setup needed',data.unavailable_reason,'warn'):''}`+
    `${data.installation_uncertain?notice('Installation needs attention','The last package installation failed. The recorded commit is the last known version; inspect the log and retry the update or recover from the terminal.','warn'):''}`+
    `<section class="panel"><div class="panel-head"><div><div class="section-kicker">SOFTWARE UPDATE</div><h2>GPU Lab ${esc(data.version)}</h2><p>Updates from ${esc(data.branch)} on GitHub</p></div><span class="pill ${data.update_available?'active':'neutral'}">${data.update_available?'Update available':'Application'}</span></div><div class="panel-body"><div class="kv"><span>Installed commit</span><code>${esc(short(data.installed_commit))}</code></div><div class="kv"><span>Latest checked commit</span><code>${esc(short(op.available_commit))}</code></div><div class="kv"><span>Last checked</span><span>${op.checked_at?esc(date(op.checked_at)):'Not checked yet'}</span></div><p>${esc(checkText)}</p><div class="actions"><button data-maintenance="check" ${disabled}>Check for updates</button><button class="primary" data-maintenance="update" ${disabled}>Update to latest</button></div><p class="inline-note">Updates wait for active benchmarks to finish. The app then briefly goes offline while the new version installs and the service restarts. Benchmark history and GPU runtimes are preserved.</p></div></section>`+
    `<section class="panel"><div class="panel-head"><div><div class="section-kicker">SERVICE</div><h2>Restart GPU Lab</h2><p>Restart the web service without downloading new code.</p></div><button data-maintenance="restart" ${disabled}>Restart service</button></div><div class="panel-body"><p class="inline-note">Restarts wait for active benchmarks. New launches pause during updates and restarts. A server access token may need to be entered again after reconnecting.</p></div></section>`+
    `<section class="panel"><div class="panel-head"><div><div class="section-kicker">MAINTENANCE ACTIVITY</div><h2>${esc(op.state.replaceAll('_',' '))}</h2></div><button data-maintenance-refresh>Refresh status</button></div><div class="panel-body"><p role="status" aria-live="polite">${esc(op.message)}</p>${op.previous_commit?`<p class="inline-note">Previous source commit: <code>${esc(short(op.previous_commit))}</code></p>`:''}${op.backup?`<p class="inline-note">Database backup: <code>${esc(op.backup)}</code></p>`:''}<h3>Latest operation log</h3><pre class="maintenance-log" tabindex="0">${esc(data.log||'No log entries yet.')}</pre>${op.state==='failed'?notice('Maintenance failed','Review the log before retrying. If the app cannot start, terminal recovery may be required.','bad'):''}</div></section>`;
  $('#app').querySelectorAll('[data-maintenance]').forEach(button=>button.onclick=()=>startMaintenance(button.dataset.maintenance));
  $('[data-maintenance-refresh]').onclick=()=>settings().catch(error=>toast(error.message));
}
async function settings(){
  if(maintenance.offline&&maintenance.pending){renderMaintenanceOffline();return}
  const data=await api('/settings');
  if(location.hash!=='#settings')return;
  renderSettings(data);
  if(data.busy&&!maintenance.pending){
    saveMaintenance({id:data.operation.id,instance:data.instance,started:Date.now()});
    maintenanceSchedule();
  }
}
async function startMaintenance(action){
  $('#app').querySelectorAll('[data-maintenance]').forEach(button=>button.disabled=true);
  try{
    const health=await fetch('/api/health',{cache:'no-store'}).then(response=>response.json());
    // Save before POST: a restart can close the connection before its response arrives.
    saveMaintenance({id:null,instance:health.instance,started:Date.now()});
    const op=await post('/settings/'+action);
    saveMaintenance({id:op.id,instance:health.instance,started:Date.now()});
    maintenance.offline=false;
    maintenanceSchedule();
    await settings();
  }catch(error){
    toast(error.message);
    if(error.status)saveMaintenance(null);
    if(maintenance.pending){renderMaintenanceOffline();maintenanceSchedule()}
    else await settings();
  }
}
if(maintenance.pending)maintenanceSchedule();
