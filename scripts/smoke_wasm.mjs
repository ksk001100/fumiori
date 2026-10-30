// Run the generated browser WASM module in Node without an HTTP server.
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import init, { Generator } from '../build/web/pkg/fumiori_wasm.js';

await init({ module_or_path: readFileSync(new URL('../build/web/pkg/fumiori_wasm_bg.wasm', import.meta.url)) });
const args = process.argv.slice(2);
assert(args[0] !== '--catalog' || args.length === 2, 'usage: --catalog path/to/catalog.json');
const models = args[0] === '--catalog' ? JSON.parse(readFileSync(args[1], 'utf8')).map(row => row.id) : args;
for (const model of models.length ? models : ['model', 'model-int8']) {
  assert.match(model, /^model(?:-[a-z0-9-]+)?$/);
  const read = name => readFileSync(new URL(`../build/web/${model}/${name}`, import.meta.url));
  const generator = new Generator(read('manifest.json'), read('model-config.json'), read('vocab.json'), read('model.safetensors'));
  const manifest = JSON.parse(read('manifest.json').toString('utf8'));
  const prompts = [];
  for (let i = 0; i < 30; i++) {
    const prompt = JSON.parse(generator.generate(['easy', 'normal', 'hard'][i % 3], 3432));
    assert.equal(prompt.source, manifest.generation_constraints ? 'model_constrained' : 'model');
    assert.equal(prompt.text, prompt.segments.map(s => s.surface).join(''));
    assert.equal(prompt.kana, prompt.segments.map(s => s.kana).join(''));
    assert.equal(prompt.text.split('、').length, prompt.kana.split('、').length);
    const [min, max] = [[5, 12], [13, 22], [23, 40]][i % 3];
    assert(prompt.kana.length >= min && prompt.kana.length <= max);
    assert(!prompt.text.endsWith('。'));
    prompts.push(prompt.text);
  }
  console.log(JSON.stringify({ model, count: prompts.length, unique: new Set(prompts).size, samples: prompts.slice(0, 6) }));
  generator.free();
}
