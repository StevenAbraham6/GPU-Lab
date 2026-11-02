const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const src=fs.readFileSync('gpulab/static/app.js','utf8');
const context=vm.createContext({clearTimeout,setTimeout,empty:(title,body)=>title+body});
vm.runInContext(src.slice(0,src.indexOf('function toast'))+src.slice(src.indexOf('const telemetrySpecs='),src.indexOf('function elapsedLabel('))+`;state.catalog={tests:[{id:'arithmetic',title:'Arithmetic'},{id:'matrix',title:'Matrix'}]};`,context);
const samples=[{ts:0,test_id:'arithmetic',power_w:50,clock_mhz:1000,utilization_pct:20,memory_used_mib:512},{ts:1,test_id:'arithmetic',power_w:null,clock_mhz:1100,utilization_pct:30,memory_used_mib:1024},{ts:2,test_id:'matrix',power_w:80,clock_mhz:1200,utilization_pct:80,memory_used_mib:2048}];
context.samples=samples;
const run=expression=>vm.runInContext(expression,context);
test('power identifies workload sections and leaves missing readings disconnected',()=>{
 const html=run('chart(samples,telemetrySpecs[0],{vram_bytes:4294967296})');
 assert.match(html,/Arithmetic/);assert.match(html,/Matrix/);assert.match(html,/estimated between samples/);
 assert.equal((html.match(/<polyline/g)||[]).length,2);
});
test('renders clock, busy and VRAM as sensor readings and handles missing capacity',()=>{
 for(const id of [1,2,3])assert.match(run(`chart(samples,telemetrySpecs[${id}],{vram_bytes:4294967296})`),/data-hit-area/);
 assert.equal(run('state.telemetryCharts.vram.rows[2].value'),50);
 assert.match(run('chart(samples,telemetrySpecs[3],{})'),/Total VRAM capacity is unavailable/);
});
test('hover uses SVG coordinates and reaches both endpoints; keyboard navigation works',()=>{
 const elements={};for(const key of ['hit-area','crosshair','point','readout','readout-bg','readout-time','readout-value','readout-workload'])elements[key]={attrs:{},events:{},setAttribute(k,v){this.attrs[k]=v},addEventListener(k,v){this.events[k]=v}};
 context.document={querySelectorAll:()=>[{dataset:{chart:'power'},querySelector:s=>elements[s.slice(6,-1)],getScreenCTM:()=>({inverse:()=>({})})}]};
 context.DOMPoint=class{constructor(x,y){this.x=x;this.y=y}matrixTransform(){return this}};
 run('chart(samples,telemetrySpecs[0],{});bindTelemetryCharts()');
 const hit=elements['hit-area'];hit.events.pointermove({clientX:55,clientY:20});assert.match(hit.attrs['aria-valuetext'],/50 W at 0 seconds/);
 hit.events.pointermove({clientX:784,clientY:20});assert.match(hit.attrs['aria-valuetext'],/80 W at 2 seconds/);
 hit.events.keydown({key:'ArrowLeft',preventDefault(){}});assert.match(hit.attrs['aria-valuetext'],/50 W at 0 seconds/);
});

test('hover resolves an original reading and ignores stale responses',async()=>{
 const elements={};for(const key of ['hit-area','crosshair','point','readout','readout-bg','readout-time','readout-value','readout-workload'])elements[key]={attrs:{},events:{},setAttribute(k,v){this.attrs[k]=v},addEventListener(k,v){this.events[k]=v}};
 context.document={querySelectorAll:()=>[{isConnected:true,dataset:{chart:'power'},querySelector:s=>elements[s.slice(6,-1)],getScreenCTM:()=>({inverse:()=>({})})}]};
 context.DOMPoint=class{constructor(x,y){this.x=x}matrixTransform(){return this}};
 let timers=[];context.setTimeout=fn=>{timers.push(fn);return timers.length};context.clearTimeout=()=>{};
 let resolve;context.api=()=>new Promise(r=>resolve=r);
 run("chart(samples,telemetrySpecs[0],{});bindTelemetryCharts('run')");
 const hit=elements['hit-area'];hit.events.pointermove({clientX:100,clientY:0});const first=timers.shift()();hit.events.pointermove({clientX:700,clientY:0});
 resolve({ts:0.5,power_w:60,test_id:'arithmetic'});await first;assert.match(elements['readout-value'].textContent,/overview/);
 const second=timers.shift()();resolve({ts:1.5,power_w:73,test_id:'matrix'});await second;assert.equal(elements['readout-value'].textContent,'73 W');assert.equal(elements['readout-workload'].textContent,'Matrix');
});
