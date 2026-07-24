"""LoRA — 低秩适应，参数高效微调"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """y = Wx + (α/r) * B * A * x — W 冻结，只训 A 和 B"""
    def __init__(self, linear: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.A = nn.Parameter(torch.empty(r, linear.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        self.B = nn.Parameter(torch.empty(linear.out_features, r))
        nn.init.zeros_(self.B)

    def forward(self, x):
        w_out = self.linear(x)
        lora_out = F.linear(F.linear(x, self.A), self.B) * self.scaling
        return w_out + lora_out

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

def apply_lora_to_model(model, r=8, alpha=16, target_names=None):
    if target_names is None:
        target_names = ['q_proj', 'k_proj', 'v_proj', 'out_proj']

    replacements = {}
    for name, module in model.named_modules():
        leaf = name.split('.')[-1]
        if leaf in target_names and isinstance(module, nn.Linear):
            replacements[name] = LoRALinear(module, r, alpha)

    for full_name, lora_module in replacements.items():
        parts = full_name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        old_module = getattr(parent, parts[-1])
        device = old_module.weight.device
        setattr(parent, parts[-1], lora_module.to(device))

    for name, param in model.named_parameters():
        if '.A' not in name and '.B' not in name:
            param.requires_grad = False

    return model


def get_lora_params(model):
    """只返回 A 和 B 的 parameters，给优化器用"""
    params = []
    for name, param in model.named_parameters():
        if '.A' in name or '.B' in name:
            params.append(param)
    return params


def get_lora_state_dict(model):
    """只保存 LoRA adaptor 权重（~1MB）"""
    state = {}
    for name, param in model.named_parameters():
        if '.A' in name or '.B' in name:
            state[name] = param.data.clone()
    return state


def load_lora_state_dict(model, state_dict):
    """加载 LoRA adaptor 权重"""
    for name, param in model.named_parameters():
        if name in state_dict:
            param.data.copy_(state_dict[name])


def merge_lora(model):
    """W = W + (α/r) * B @ A，之后推理无 LoRA 开销"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            delta = (module.scaling * module.B.data @ module.A.data).to(
                module.linear.weight.dtype
            )
            module.linear.weight.data += delta


def unmerge_lora(model):
    """W = W - (α/r) * B @ A，撤销 merge"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            delta = (module.scaling * module.B.data @ module.A.data).to(
                module.linear.weight.dtype
            )
            module.linear.weight.data -= delta
