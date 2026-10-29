"""Block-128 symmetric INT4 fake quantization with straight-through gradients."""
import torch
from torch import nn
from torch.nn.utils import parametrize


def fake_int4(weight):
    if weight.ndim!=2 or weight.numel()%128:
        raise ValueError('INT4 QAT needs block-aligned matrices')
    blocks=weight.reshape(-1,128)
    scale=blocks.detach().abs().amax(1,keepdim=True).clamp_min(1e-12)/7
    dequantized=(blocks.detach()/scale).round().clamp(-7,7)*scale
    return weight+(dequantized.reshape_as(weight)-weight).detach()


class BlockInt4(nn.Module):
    def forward(self,weight):
        return fake_int4(weight)


def enable_int4(model):
    for module in list(model.modules()):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            parametrize.register_parametrization(module,'weight',BlockInt4())


def plain_state(model):
    return {name.replace('.parametrizations.weight.original','.weight'):value.detach().cpu()
            for name,value in model.state_dict().items()}
