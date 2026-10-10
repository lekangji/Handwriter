const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source=fs.readFileSync('webapp/static/jobs.js','utf8');
(async()=>{
 const states=[{status:'queued',progress:0},{status:'generating',progress:50},{status:'done',progress:100,result:{num_pages:2}}];
 let active=0,maxActive=0;
 const context=vm.createContext({window:{},encodeURIComponent,setTimeout:fn=>fn(),fetch:async()=>{
  active++; maxActive=Math.max(active,maxActive);
  const state=states.shift(); active--; return {ok:true,json:async()=>state};
 }});
 vm.runInContext(source,context);
 const progress=[];
 const result=await context.window.HandwriterJobs.wait('test',state=>progress.push(state.progress));
 assert.equal(result.num_pages,2); assert.deepEqual(progress,[0,50,100]); assert.equal(maxActive,1);
 context.fetch=async()=>({ok:false,json:async()=>({error:'Bad input'})});
 await assert.rejects(context.window.HandwriterJobs.start('/generate',{}),/Bad input/);
 context.fetch=async()=>({ok:true,json:async()=>({status:'error',error:'Unreadable text'})});
 await assert.rejects(context.window.HandwriterJobs.wait('bad',()=>{}),/Unreadable text/);
 context.fetch=async()=>{throw new Error('Network failure')};
 await assert.rejects(context.window.HandwriterJobs.wait('lost',()=>{}),/Connection lost/);
 console.log('Job checks passed: serial progress polling, result, start failure, generation failure, connection loss.');
})().catch(error=>{console.error(error);process.exitCode=1});
