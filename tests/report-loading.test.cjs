const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync('gpulab/static/app.js','utf8');
test('shows loading feedback before the summary request finishes',()=>{
 const app={innerHTML:'previous screen'},state={reportGeneration:0};let requested;
 const context=vm.createContext({state,$:()=>app,api:path=>{requested=path;return new Promise(()=>{})}});
 vm.runInContext(source.slice(source.indexOf('async function detail('),source.indexOf('async function compare(')),context);
 context.detail('run-id');
 assert.match(app.innerHTML,/Opening benchmark report/);
 assert.equal(requested,'/runs/run-id?view=overview');
});
test('shows the report shell without waiting for results or telemetry',async()=>{
 const app={innerHTML:''},state={reportGeneration:0,catalog:{tests:[]},timer:null},requests=[];
 const run={results:[],tests:['matrix'],gpu:{name:'GPU',id:'gpu'},metadata:{name:'Old run',description:'',tags:[]},status:'completed',revision:1,snapshot:{},config:{},audit:[],created:'today'};
 const context=vm.createContext({state,location:{hash:'#run/test'},document:{querySelectorAll:()=>[]},$:()=>app,api:path=>{requests.push(path);return path.endsWith('view=overview')?Promise.resolve(run):new Promise(()=>{})},head:()=>'<h1>REPORT</h1>',notice:()=>'',stat:(a,b)=>`<div class="value">${b}</div><div class="sub"></div>`,active:()=>false,esc:String,json:()=>'',badge:()=>'',demo:()=>'',dl:()=>'',glance:()=>'',telemetrySpecs:[],empty:()=>''});
 vm.runInContext(source.slice(source.indexOf('async function detail('),source.indexOf('async function compare(')),context);
 await context.detail('test');
 assert.match(app.innerHTML,/REPORT/);assert.match(app.innerHTML,/Loading benchmark results/);
 assert.ok(requests.includes('/runs/test?view=summary'));assert.ok(requests.includes('/runs/test/telemetry'));
});
