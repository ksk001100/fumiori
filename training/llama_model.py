"""Small Llama-compatible body, preserving source RMSNorm/RoPE/SwiGLU weights."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self,x):
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+self.eps)
        return normalized.to(x.dtype)*self.weight


def rotary(x, theta):
    t,d = x.shape[-2:]
    inv = 1.0 / theta ** (torch.arange(0,d,2,device=x.device,dtype=torch.float32)/d)
    angles = torch.outer(torch.arange(t,device=x.device,dtype=torch.float32),inv)
    angles = torch.cat((angles,angles),dim=-1)[None,None]
    half = torch.cat((-x[...,d//2:],x[...,:d//2]),dim=-1)
    return x*angles.cos().to(x.dtype)+half*angles.sin().to(x.dtype)


class Attention(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.c = c
        self.q_proj = nn.Linear(c.d_model,c.d_model,bias=False)
        self.k_proj = nn.Linear(c.d_model,c.d_model,bias=False)
        self.v_proj = nn.Linear(c.d_model,c.d_model,bias=False)
        self.o_proj = nn.Linear(c.d_model,c.d_model,bias=False)

    def forward(self,x):
        b,t,d=x.shape
        def heads(project):
            return project(x).reshape(b,t,self.c.n_heads,d//self.c.n_heads).transpose(1,2)
        q,k,v=heads(self.q_proj),heads(self.k_proj),heads(self.v_proj)
        q,k=rotary(q,self.c.rope_theta),rotary(k,self.c.rope_theta)
        scores=q@k.transpose(-2,-1)/math.sqrt(d//self.c.n_heads)
        mask=torch.ones(t,t,device=x.device,dtype=torch.bool).triu(1)
        scores=scores.masked_fill(mask,-torch.inf)
        z=(scores.float().softmax(-1).to(v.dtype)@v).transpose(1,2).reshape(b,t,d)
        return self.o_proj(z)


class MLP(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.gate_proj=nn.Linear(c.d_model,c.d_ff,bias=False)
        self.up_proj=nn.Linear(c.d_model,c.d_ff,bias=False)
        self.down_proj=nn.Linear(c.d_ff,c.d_model,bias=False)

    def forward(self,x):
        return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))


class LlamaBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.input_layernorm=RMSNorm(c.d_model,c.norm_eps)
        self.post_attention_layernorm=RMSNorm(c.d_model,c.norm_eps)
        self.self_attn=Attention(c)
        self.mlp=MLP(c)

    def forward(self,x):
        x=x+self.self_attn(self.input_layernorm(x))
        return x+self.mlp(self.post_attention_layernorm(x))


class LlamaTransformer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.embed_tokens=nn.Embedding(config.vocab_size,config.d_model)
        self.layers=nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layers)])
        self.norm=RMSNorm(config.d_model,config.norm_eps)
        self.lm_head=nn.Linear(config.d_model,config.vocab_size,bias=False)

    def forward(self,ids):
        if ids.ndim!=2 or not 0<ids.shape[1]<=self.config.max_seq_len:
            raise ValueError('invalid sequence shape/length')
        x=self.embed_tokens(ids)
        for layer in self.layers:
            x=layer(x)
        return self.lm_head(self.norm(x))

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
