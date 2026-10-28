// Run with Node.js and Playwright available (NODE_PATH may point at installed packages).
const { chromium, firefox, webkit } = require('playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const http = require('node:http');
const path = require('node:path');
(async () => {
  const root = path.resolve('build/web');
  const server = http.createServer((req,res)=>{
    const pathname = decodeURIComponent(new URL(req.url,'http://localhost').pathname);
    const file = path.resolve(root, '.' + (pathname==='/'?'/index.html':pathname));
    if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
    fs.readFile(file,(err,data)=>{
      if(err) { res.writeHead(404).end(); return; }
      res.setHeader('Content-Type', ({'.js':'text/javascript','.wasm':'application/wasm','.html':'text/html; charset=utf-8','.json':'application/json'})[path.extname(file)] || 'application/octet-stream');
      res.end(data);
    });
  });
  await new Promise(resolve=>server.listen(8001,'127.0.0.1',resolve));
  const results = [];
  for (const [name, engine] of Object.entries({chromium, firefox, webkit})) {
    let browser;
    try { browser = await engine.launch({headless:true, ...(name==='chromium'?{channel:'msedge'}:{})}); }
    catch (e) { results.push({browser:name, status:'unavailable', reason:String(e).split('\n')[0]}); continue; }
    try {
      const page = await browser.newPage();
      await page.goto('http://127.0.0.1:8001');
      const measurements = await page.evaluate(async () => {
        const worker = new Worker('./worker.js', {type:'module'});
        function request(data) {
          return new Promise((resolve,reject) => {
            const timer = setTimeout(()=>reject(Error('Worker timeout')),120000);
            worker.onmessage = ({data}) => { clearTimeout(timer); data.command==='error'?reject(Error(data.message)):resolve(data); };
            worker.onerror = e => { clearTimeout(timer); reject(Error(e.message)); };
            worker.postMessage(data);
          });
        }
        const results=[];
        try {
          for(const model of [...document.querySelector('#model').options].map(option=>option.value)) {
            const load=await request({command:'load',model});
            const fixtures=await (await fetch(`./${model}/parity.json`)).json();
            let maxError=0;
            for(const fixture of fixtures) {
              const actual=await request({command:'logits',ids:fixture.ids});
              if(actual.logits.length!==fixture.logits.length) throw Error('Sequence shape mismatch');
              fixture.logits.forEach((row,i)=>{
                if(actual.logits[i].length!==row.length) throw Error('Vocab shape mismatch');
                row.forEach((v,j)=>{ const error=Math.abs(v-actual.logits[i][j]); if(!Number.isFinite(error)) throw Error('Nonfinite logits'); maxError=Math.max(maxError,error); });
              });
            }
            const samples=[];
            for(let i=0;i<12;i++) samples.push(await request({command:'generate',difficulty:['easy','normal','hard'][i%3],seed:42}));
            results.push({model,load,max_logit_error:maxError,samples});
          }
          let error;
          try { await request({command:'load',model:'missing-model'}); } catch(e) { error=String(e); }
          if(!error) throw Error('Missing bundle accepted');
          return {models:results, missing_model_error:error, userAgent:navigator.userAgent};
        } finally { worker.terminate(); }
      });
      for (const model of measurements.models) {
        assert(model.max_logit_error < 2e-4);
        for (const {prompt:p} of model.samples) {
          assert.equal(p.text,p.segments.map(s=>s.surface).join(''));
          assert.equal(p.kana,p.segments.map(s=>s.kana).join(''));
          assert.match(p.kana,/^[ぁ-ゖー]+$/u);
          assert(['model','model_constrained'].includes(p.source));
          assert(!p.text.endsWith('。'));
        }
      }
      await page.click('#load');
      await page.waitForFunction(()=>!document.querySelector('#generate').disabled,null,{timeout:120000});
      await page.click('#generate');
      await page.waitForFunction(()=>document.querySelector('#source').textContent.startsWith('モデル生成'),null,{timeout:120000});
      if (measurements.models.length > 1) {
        await page.selectOption('#model',measurements.models.at(-1).model);
        await page.click('#load');
        await page.waitForFunction(()=>!document.querySelector('#generate').disabled,null,{timeout:120000});
        await page.click('#generate');
        await page.waitForFunction(()=>document.querySelector('#source').textContent.startsWith('モデル生成'),null,{timeout:120000});
      }
      await page.screenshot({path:`build/web-${name}.png`,fullPage:true});
      results.push({browser:name,version:browser.version(),status:'passed',...measurements});
    } catch(e) { results.push({browser:name,status:'failed',error:String(e)}); }
    finally { await browser.close(); }
  }
  fs.writeFileSync(process.env.FUMIORI_BROWSER_REPORT || 'docs/wasm-measurements.json',JSON.stringify({measured_at:new Date().toISOString(),results},null,2));
  console.log(JSON.stringify(results.map(({browser,status,error})=>({browser,status,error}))));
  if(!results.some(r=>r.status==='passed') || results.some(r=>r.status==='failed')) process.exitCode=1;
  server.close();
})();
