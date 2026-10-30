const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

function page(overrides={}){
  const storage=new Map(),app={innerHTML:'',querySelectorAll:()=>[]};
  const context=vm.createContext({
    console,Date,JSON,AbortSignal,
    sessionStorage:{getItem:key=>storage.get(key),setItem:(key,value)=>storage.set(key,value),removeItem:key=>storage.delete(key)},
    setTimeout:()=>1,clearTimeout:()=>{},
    location:{hash:'#settings',reload:()=>context.reloaded=true},
    $:()=>app,head:()=>'',notice:(title,body)=>title+' '+body,esc:String,date:String,toast:()=>{},
    fetch:async()=>({ok:true,json:async()=>({instance:'old'})}),
    api:async()=>({busy:false,operation:{id:'job',state:'completed'}}),
    post:async()=>({id:'job'}),
    ...overrides,
  });
  vm.runInContext(fs.readFileSync('gpulab/static/settings.js','utf8'),context);
  vm.runInContext('renderSettings=()=>{}',context);
  return {context,app,storage,run:code=>vm.runInContext(code,context)};
}

test('reloads after the server instance changes and clears the pending operation',async()=>{
  const p=page({fetch:async()=>({ok:true,json:async()=>({instance:'new'})})});
  p.run("saveMaintenance({id:'job',instance:'old',started:Date.now()})");
  await p.run('pollMaintenance()');
  assert.equal(p.context.reloaded,true);
  assert.equal(p.storage.size,0);
});

test('a dropped POST response keeps polling because the update may have started',async()=>{
  const p=page({post:async()=>{throw new TypeError('Connection closed')}});
  await p.run("startMaintenance('restart')");
  assert.equal(p.run('maintenance.pending.instance'),'old');
  assert.match(p.app.innerHTML,/Waiting for the server/);
});

test('an explicit rejected POST clears reconnect state',async()=>{
  const p=page({post:async()=>{const error=new Error('Already busy');error.status=409;throw error}});
  p.run('settings=async()=>{}');
  await p.run("startMaintenance('restart')");
  assert.equal(p.run('maintenance.pending'),null);
});

test('a successful check completes without requiring a restart',async()=>{
  const p=page();
  p.run("saveMaintenance({id:'job',instance:'old',started:Date.now()})");
  await p.run('pollMaintenance()');
  assert.equal(p.run('maintenance.pending'),null);
  assert.equal(p.context.reloaded,undefined);
});

test('temporary downtime keeps the operation and shows a reconnect message',async()=>{
  const p=page({fetch:async()=>{throw new TypeError('Offline')}});
  p.run("saveMaintenance({id:'job',instance:'old',started:Date.now()})");
  await p.run('pollMaintenance()');
  assert.equal(p.run('maintenance.offline'),true);
  assert.equal(p.run('maintenance.pending.id'),'job');
  assert.match(p.app.innerHTML,/reconnect and reload automatically/);
});

test('reconnection can resume after refreshing the browser tab',()=>{
  const pending=JSON.stringify({id:'saved',instance:'old',started:Date.now()});
  const p=page({sessionStorage:{getItem:()=>pending}});
  assert.equal(p.run('maintenance.pending.id'),'saved');
});
