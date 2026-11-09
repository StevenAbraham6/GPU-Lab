const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync('gpulab/static/app.js','utf8');
const prefix=source.slice(0,source.indexOf('function toast'));
const config=source.slice(source.indexOf('const aiGroups='),source.indexOf('function newPayload()'));
const result=source.slice(source.indexOf('function aiResults('),source.indexOf('function caseHtml('));
const context=vm.createContext({});
vm.runInContext(prefix+config+result,context);

test('AI controls expose independent batching, baseline and lengths',()=>{
  const html=context.aiOptionsHtml('ai_moe','MoE');
  assert.match(html,/name="ai_moe_batch_sizes" value="8"/);
  assert.match(html,/Existing baseline/);
  assert.match(html,/value="2048"/);
  assert.match(html,/value="512"/);
  assert.equal(context.aiCaseCount({batch_sizes:[1,4],input_lengths:['baseline',128],output_lengths:[64,256]}),8);
});

test('AI case table distinguishes throughput and failed cases',()=>{
  const html=context.aiResults({test_id:'ai_moe',cases:[{name:'case',status:'passed',settings:{batch_size:4,input_length:128,max_output_tokens:64},metrics:[{name:'output tokens per second',value:100,unit:'tokens/s'},{name:'per-request output speed',value:25,unit:'tokens/s'}]},{name:'oom',status:'error',reason:'out of memory',settings:{batch_size:8,input_length:128,max_output_tokens:64},metrics:[]}]});
  assert.match(html,/Total speed/);
  assert.match(html,/100 tokens\/s/);
  assert.match(html,/25 tokens\/s/);
  assert.match(html,/out of memory/);
  assert.match(html,/data-ai-chart-input/);
});
