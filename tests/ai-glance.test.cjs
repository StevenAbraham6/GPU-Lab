const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync('gpulab/static/app.js','utf8');
const context=vm.createContext({});
vm.runInContext(source.slice(0,source.indexOf('function toast'))+source.slice(source.indexOf('function representative('),source.indexOf('function seriesChart(')),context);
vm.runInContext("state.catalog={tests:[{id:'ai',title:'AI inference stability'}]}",context);
const render=metrics=>context.glance({results:[{test_id:'ai',status:'passed',cases:[{name:'Qwen',status:'passed',metrics}]}]});
test('shows AI token speed even when completed requests is the first metric',()=>{
  const html=render([{name:'completed requests',value:99,unit:'requests',direction:'context'},{name:'output tokens per second',value:42,unit:'tokens/s',direction:'higher'}]);
  assert.match(html,/42<\/span><span class="unit">tokens\/s/);
  assert.doesNotMatch(html,/>99<\/span>/);
});
test('does not mislabel requests as speed if token speed is absent',()=>{
  assert.match(render([{name:'completed requests',value:99,unit:'requests',direction:'context'}]),/No valid measurement/);
});
test('supports the existing simulated AI throughput metric',()=>{
  assert.match(render([{name:'throughput',value:42,unit:'tokens/s',direction:'higher'}]),/tokens\/s/);
});
