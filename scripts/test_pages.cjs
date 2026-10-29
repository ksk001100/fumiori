// Smoke-test the real module Worker and relative URLs beneath a Pages project path.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const http = require('node:http');
const path = require('node:path');

(async () => {
  const root = path.resolve(process.env.FUMIORI_WEB_ROOT || 'build/web');
  const prefix = '/fumiori/';
  const server = http.createServer(async (req, res) => {
    const pathname = new URL(req.url, 'http://localhost').pathname;
    if (!pathname.startsWith(prefix)) { res.writeHead(404).end(); return; }
    const file = path.resolve(root, pathname.slice(prefix.length) || 'index.html');
    if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
    try {
      const data = await fs.readFile(file);
      res.setHeader('Content-Type', ({ '.js': 'text/javascript', '.wasm': 'application/wasm',
        '.html': 'text/html; charset=utf-8', '.json': 'application/json' })[path.extname(file)] || 'application/octet-stream');
      res.end(data);
    } catch { res.writeHead(404).end(); }
  });
  let browser;
  try {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    browser = await chromium.launch({ headless: true,
      ...(process.env.FUMIORI_BROWSER_CHANNEL ? { channel: process.env.FUMIORI_BROWSER_CHANNEL } : {}) });
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    page.on('response', response => { if (!response.ok()) errors.push(`${response.status()} ${response.url()}`); });
    await page.goto(`http://127.0.0.1:${server.address().port}${prefix}`);
    const html = await fs.readFile(path.join(root, 'index.html'), 'utf8');
    assert.match(html, /new Worker\('\.\/worker\.js\?v=[a-f0-9]{20}'/);
    const models = await page.locator('#model option').evaluateAll(options => options.map(o => o.value));
    if (process.env.FUMIORI_REQUIRE_EXPERIMENTAL) assert(models.includes('model-experimental'));
    const visits = models.length > 1 ? [...models, models[0]] : models;
    const timings = [];
    for (const model of visits) {
      await page.selectOption('#model', model);
      assert(await page.isDisabled('#generate'));
      assert.equal(await page.textContent('#kana'), '');
      if (model === 'model-experimental') assert.match(await page.textContent('#model-info'), /試作/);
      await page.click('#load');
      await page.waitForFunction(() => !document.querySelector('#generate').disabled, null, { timeout: 120000 });
      const loaded = JSON.parse(await page.textContent('#metrics'));
      assert.equal(loaded.model, model);
      const files = ['manifest.json', 'model-config.json', 'vocab.json', 'model.safetensors'];
      const sizes = await Promise.all(files.map(async name => (await fs.stat(path.join(root, model, name))).size));
      assert.equal(loaded.bundle_bytes, sizes.reduce((sum, size) => sum + size, 0));
      for (const difficulty of ['easy', 'normal', 'hard']) {
        await page.selectOption('#difficulty', difficulty);
        await page.click('#generate');
        await page.waitForFunction(() => !document.querySelector('#generate').disabled, null, { timeout: 120000 });
        assert.match(await page.textContent('#source'), /^モデル生成/);
        assert.match(await page.textContent('#kana'), /^[ぁ-ゖー、]+$/u);
        assert.equal(await page.textContent('#status'), '準備完了');
        assert.equal(await page.textContent('#active-model'), await page.locator('#model option:checked').textContent());
      }
      const count = Number(process.env.FUMIORI_TIMING_SAMPLES || 0);
      if (count > 0 && !timings.some(r => r.model === model)) {
        await page.selectOption('#difficulty', 'normal');
        const ms = [];
        for (let i = 0; i < count; i++) {
          await page.click('#generate');
          await page.waitForFunction(() => !document.querySelector('#generate').disabled, null, { timeout: 120000 });
          const result = JSON.parse(await page.textContent('#metrics'));
          assert.equal(result.command, 'generated');
          ms.push(result.generation_ms);
        }
        const sorted = [...ms].sort((a,b) => a-b);
        const manifest = JSON.parse(await fs.readFile(path.join(root, model, 'manifest.json'), 'utf8'));
        timings.push({model, samples:count, difficulty:'normal', seed:42, generation_defaults:manifest.generation_defaults,
          weights_sha256:manifest.weights_sha256, p50_ms:sorted[Math.floor((count-1)*.5)],
          p95_ms:sorted[Math.ceil((count-1)*.95)], all_ms:ms});
      }
    }
    assert.deepEqual(errors, []);
    // An old cached Worker used to report ready without identifying the model.
    // The UI must explain the mismatch instead of leaving a silent disabled button.
    const stale = await browser.newPage();
    await stale.route('**/worker.js*', route => route.fulfill({ contentType: 'text/javascript',
      body: "self.onmessage=()=>self.postMessage({command:'loaded',bundle_bytes:1});" }));
    await stale.goto(`http://127.0.0.1:${server.address().port}${prefix}`);
    await stale.click('#load');
    await stale.waitForFunction(() => document.querySelector('#status').textContent.includes('再読み込み'));
    assert(await stale.isDisabled('#generate'));
    assert(!(await stale.isDisabled('#load')));
    await stale.close();
    if (timings.length) {
      const result = {browser:browser.version(), measurement:'Web Worker generation_ms; model load excluded; configured history retained',timings};
      console.log(JSON.stringify(result));
      if (process.env.FUMIORI_TIMING_REPORT) await fs.writeFile(process.env.FUMIORI_TIMING_REPORT, JSON.stringify(result,null,2)+'\n');
    }
    console.log(`Pages project path: ${visits.join(' -> ')}; all three difficulties passed`);
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
