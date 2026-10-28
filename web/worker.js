import init, { Generator } from './pkg/fumiori_wasm.js';
let generator;
let memory;
self.onmessage = async ({data}) => {
  try {
    if (data.command === 'load') {
      generator?.free();
      generator = undefined;
      const start = performance.now();
      const wasm = await init();
      memory = wasm.memory;
      const names = ['manifest.json', 'model-config.json', 'vocab.json', 'model.safetensors'];
      const files = await Promise.all(names.map(async name => {
        const response = await fetch(`./${data.model}/${name}`);
        if (!response.ok) throw Error(`${name}: HTTP ${response.status}`);
        return new Uint8Array(await response.arrayBuffer());
      }));
      generator = new Generator(...files);
      self.postMessage({command:'loaded', load_ms:performance.now()-start,
        wasm_linear_memory_bytes:memory.buffer.byteLength, bundle_bytes:files.reduce((n,b)=>n+b.length,0)});
    } else if (data.command === 'logits') {
      if (!generator) throw Error('Load a model first');
      self.postMessage({command:'logits', logits:JSON.parse(generator.logits(new Uint32Array(data.ids)))});
    } else if (data.command === 'generate') {
      if (!generator) throw Error('Load a model first');
      const start = performance.now();
      const prompt = JSON.parse(generator.generate(data.difficulty, data.seed));
      self.postMessage({command:'generated', prompt, generation_ms:performance.now()-start,
        wasm_linear_memory_bytes:memory.buffer.byteLength});
    } else throw Error('Unknown command');
  } catch(error) {
    self.postMessage({command:'error', message:String(error)});
  }
};
