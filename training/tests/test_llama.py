import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import torch
from safetensors.torch import load_file,save_file
from training.model import ModelConfig,make_model
from training.export import export
from training.quantize import quantize
from training.package import package


def run_cli(*args):
    return subprocess.run(['cargo','run','--quiet','-p','fumiori-cli','--',*map(str,args)],capture_output=True,encoding='utf-8',timeout=180)


@pytest.fixture
def llama_export(tmp_path):
    torch.manual_seed(19)
    torch.set_num_threads(2)
    c=ModelConfig(vocab_size=16,d_model=16,n_heads=2,n_layers=2,d_ff=32,max_seq_len=16,architecture='llama')
    model=make_model(c).eval()
    vocab=json.loads(Path('data/paired-v1/vocab.json').read_text(encoding='utf-8'))
    vocab['tokens']=vocab['tokens'][:16]
    payload=json.dumps(vocab,ensure_ascii=False).encode()
    ck=dict(state_dict=model.state_dict(),model_config=vars(c),vocab_bytes=payload,
            data_manifest=dict(splits=dict(train=1)),config=dict(seed=19),torch_version=str(torch.__version__))
    path=tmp_path/'llama.pt'
    torch.save(ck,path)
    output=tmp_path/'export'
    export(path,output)
    return output


def test_llama_causal_and_full_precision_native_parity(llama_export):
    from training.train import load_checkpoint
    model,_=load_checkpoint(llama_export.parent/'llama.pt')
    with torch.no_grad():
        a=model(torch.tensor([[0,3,7,8]]));b=model(torch.tensor([[0,3,7,9]]))
    torch.testing.assert_close(a[:,:3],b[:,:3],atol=0,rtol=0)
    result=run_cli('parity','--model',llama_export)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['passed']


@pytest.mark.parametrize('bits',[4,8])
def test_llama_quantized_native_parity_and_package(llama_export,tmp_path,bits):
    target=tmp_path/f'q{bits}'
    quantize(llama_export,target,bits)
    result=run_cli('parity','--model',target)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['passed']
    runtime=tmp_path/f'runtime{bits}'
    package(target,runtime)
    assert run_cli('logits','--model',runtime,'--ids','0,3,7').stdout==run_cli('logits','--model',target,'--ids','0,3,7').stdout
    if bits==4:
        weights={k:v.clone() for k,v in load_file(target/'model.safetensors').items()}
        assert weights['embed_tokens.weight'].dtype==torch.uint8
        assert weights['embed_tokens.weight'].numel()==16*16//2
        weights['embed_tokens.weight.scale'][0]=0
        save_file(weights,target/'model.safetensors')
        path=target/'manifest.json';manifest=json.loads(path.read_text())
        manifest['weights_sha256']=hashlib.sha256((target/'model.safetensors').read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest))
        assert 'invalid INT4 scale' in run_cli('logits','--model',target,'--ids','0,3').stderr


def test_llama_matches_huggingface_reference():
    transformers=pytest.importorskip('transformers')
    torch.manual_seed(91)
    torch.set_num_threads(2)
    c=ModelConfig(vocab_size=16,d_model=16,n_heads=2,n_layers=2,d_ff=32,max_seq_len=16,architecture='llama')
    model=make_model(c).eval()
    hf=transformers.LlamaForCausalLM(transformers.LlamaConfig(vocab_size=16,hidden_size=16,
        num_hidden_layers=2,intermediate_size=32,num_attention_heads=2,num_key_value_heads=2,
        rms_norm_eps=c.norm_eps,rope_theta=c.rope_theta,attention_bias=False,mlp_bias=False,
        tie_word_embeddings=False,attention_dropout=0.,_attn_implementation='eager')).eval()
    hf.load_state_dict({('model.'+k if k!='lm_head.weight' else k):v for k,v in model.state_dict().items()})
    ids=torch.tensor([[0,3,7,8,9]])
    with torch.no_grad():
        torch.testing.assert_close(model(ids),hf(ids).logits,atol=2e-6,rtol=2e-5)


def test_qat_matches_int4_grid_and_preserves_gradients():
    from training.qat import fake_int4,enable_int4,plain_state
    torch.manual_seed(20)
    weight=torch.randn(16,16,requires_grad=True)
    quantized=fake_int4(weight)
    blocks=weight.detach().reshape(-1,128)
    scale=blocks.abs().amax(1,keepdim=True)/7
    expected=(blocks/scale).round().clamp(-7,7)*scale
    torch.testing.assert_close(quantized,expected.reshape_as(weight))
    quantized.sum().backward()
    assert torch.equal(weight.grad,torch.ones_like(weight))
    c=ModelConfig(vocab_size=16,d_model=16,n_heads=2,n_layers=2,d_ff=32,architecture='llama')
    model=make_model(c)
    original={k:v.clone() for k,v in model.state_dict().items()}
    enable_int4(model)
    restored=make_model(c)
    restored.load_state_dict(plain_state(model))
    for name,value in restored.state_dict().items():
        torch.testing.assert_close(value,original[name],atol=0,rtol=0)


def test_cli_relative_parity_tolerance_is_explicit(llama_export):
    path=llama_export/'parity.json'
    cases=json.loads(path.read_text())
    value=cases[0]['logits'][0][0]
    cases[0]['logits'][0][0]+=max(abs(value),.01)*.0005
    path.write_text(json.dumps(cases))
    strict=run_cli('parity','--model',llama_export,'--tolerance','0.000001')
    assert strict.returncode!=0
    relative=run_cli('parity','--model',llama_export,'--tolerance','0.000001','--relative-tolerance','0.001')
    assert relative.returncode==0,relative.stderr
