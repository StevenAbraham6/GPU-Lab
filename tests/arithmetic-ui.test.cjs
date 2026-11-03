const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

const source=fs.readFileSync('gpulab/static/app.js','utf8');
const context=vm.createContext({});
vm.runInContext(source.slice(0,source.indexOf('function toast'))+
  source.slice(source.indexOf('function arithmeticHtml('),source.indexOf('function caseHtml(')),context);
const render=cases=>context.arithmeticHtml({cases});
const sample=(n,power=50)=>({name:`FP32 queued · ${n} accumulators`,status:'passed',settings:{independent_accumulators:n},metrics:[
  {name:'throughput',value:10000+n,unit:'GFLOP/s'},
  {name:'elapsed throughput',value:9000,unit:'GFLOP/s'},
  ...(power===null?[]:[{name:'average GPU power',value:power,unit:'W'}]),
  {name:'work per joule',value:100,unit:'GFLOP/J'},
]});

test('renders one FP32 table with all accumulator levels and power, hiding old energy and INT32',()=>{
  const html=render([sample(32),sample(4),sample(8),sample(16),{name:'INT32',status:'passed',metrics:[]}]);
  assert.match(html,/<h3>FP32<\/h3>/);
  assert.match(html,/Avg GPU power/);
  assert.ok(html.indexOf('4 accumulators')<html.indexOf('8 accumulators'));
  assert.ok(html.indexOf('16 accumulators')<html.indexOf('32 accumulators'));
  assert.match(html,/50 W/);
  assert.doesNotMatch(html,/INT32|work per joule|energy for measured work|energy per work unit/);
});

test('handles missing power and a historical baseline',()=>{
  assert.match(render([sample(4,null)]),/Unavailable/);
  assert.match(render([{name:'FP32',status:'passed',settings:{},metrics:[{name:'throughput',value:9000,unit:'GFLOP/s'}]}]),/Original baseline/);
});
