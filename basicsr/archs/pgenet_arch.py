import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from einops.layers.torch import Rearrange
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import ResidualBlocksWithInputConv, PixelShufflePack, flow_warp
from basicsr.archs.spynet_arch import SpyNet

from mmcv.ops.deform_conv import DeformConv2dFunction
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, modulated_deform_conv2d

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)



class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class HighPassConv2d(nn.Module):
    def __init__(self, c, freeze=True):
        super().__init__()
        self.conv = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False, groups=c)
        kernel = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=torch.float32)
        self.conv.weight.data = kernel.repeat(c, 1, 1, 1)
        if freeze:
            self.conv.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)

class FrequencyEmbedding(nn.Module):
    def __init__(self, dim):
        super(FrequencyEmbedding, self).__init__()
        self.high_conv = nn.Sequential(
            HighPassConv2d(dim, freeze=True),
            nn.GELU())
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim)
        )

    def forward(self, x):
        x = self.high_conv(x)
        x = x.mean(dim=(-2, -1))
        x = self.mlp(x)
        return x

class RoutingFunction(nn.Module):
    def __init__(self, dim, freq_dim, num_experts, k, complexity, use_complexity_bias=True, complexity_scale="max"):
        super(RoutingFunction, self).__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(dim, num_experts, bias=False)
        )
        # Ensure the linear layer has the correct input dimension
        self.freq_gate = nn.Linear(freq_dim, num_experts, bias=False)
        if complexity_scale == "min":
            complexity = complexity / complexity.min()
        elif complexity_scale == "max":
            complexity = complexity / complexity.max()
        self.register_buffer('complexity', complexity)
        self.k = k
        self.tau = 1
        self.num_experts = num_experts
        self.noise_std = (1.0 / num_experts) * 1.0
        self.use_complexity_bias = use_complexity_bias

    def forward(self, x, freq_emb):
        # if freq_emb.dim() > 2:
        #     freq_emb = freq_emb.squeeze()
        logits = self.gate(x) + self.freq_gate(freq_emb)
        if self.training:
            noise = torch.randn_like(logits) * self.noise_std
            noisy_logits = logits + noise
            gating_scores = noisy_logits.softmax(dim=-1)
            top_k_values, top_k_indices = torch.topk(gating_scores, self.k, dim=-1)
        else:
            gating_scores = logits.softmax(dim=-1)
            top_k_values, top_k_indices = torch.topk(gating_scores, self.k, dim=-1)
        gates = torch.zeros_like(logits).scatter_(1, top_k_indices, top_k_values)
        return gates, top_k_indices, top_k_values

class ModExpert(nn.Module):
    def __init__(self, dim, rank, func, depth, patch_size, kernel_size):
        super(ModExpert, self).__init__()
        self.depth = depth
        self.proj = nn.ModuleList([
            nn.Conv2d(dim, rank, kernel_size=1, bias=False),
            nn.Conv2d(dim, rank, kernel_size=1, bias=False),
            nn.Conv2d(rank, dim, kernel_size=1, bias=False)
        ])
        self.body = func(rank, kernel_size=kernel_size, patch_size=patch_size)

    def forward(self, x, shared):
        shortcut = x
        x = self.proj[0](x)
        x = self.body(x) * F.silu(self.proj[1](shared))
        x = self.proj[2](x)
        return x + shortcut

class AdapterLayer(nn.Module):
    def __init__(self, dim, rank, num_experts=4, top_k=2, expert_layer=nn.Identity, stage_depth=1,
                 depth_type="lin", rank_type="constant", freq_dim=128, with_complexity=False, complexity_scale="min"):
        super().__init__()
        self.tau = 1
        self.loss = None
        self.top_k = top_k
        self.num_experts = num_experts
        self.freq_embed = FrequencyEmbedding(dim)  
        patch_sizes = [2**(i+2) for i in range(num_experts)]
        kernel_sizes = [3 + 2*i for i in range(num_experts)]
        if depth_type == "lin":
            depths = [stage_depth + i for i in range(num_experts)]
        elif depth_type == "constant":
            depths = [stage_depth for _ in range(num_experts)]
        else:
            raise NotImplementedError
        if rank_type == "constant":
            ranks = [rank for _ in range(num_experts)]
        elif rank_type == "spread":
            ranks = [dim // (2**i) for i in range(num_experts)][::-1]
        else:
            raise NotImplementedError
        self.experts = nn.ModuleList([
            ModExpert(dim, rank=r, func=expert_layer, depth=d, patch_size=p, kernel_size=k)
            for d, r, p, k in zip(depths, ranks, patch_sizes, kernel_sizes)
        ])
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        expert_complexity = torch.tensor([sum(p.numel() for p in expert.parameters()) for expert in self.experts])
        self.routing = RoutingFunction(
            dim, dim, num_experts, top_k, expert_complexity, 
            use_complexity_bias=with_complexity, complexity_scale=complexity_scale
        )

    def forward(self, x, shared):
        freq_emb = self.freq_embed(x)  # shape: [B, dim]
        gates, top_k_indices, top_k_values = self.routing(x, freq_emb)
        if self.training:
            # Sparse dispatch for training
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x)
            expert_shared_inputs = dispatcher.dispatch(shared)
            expert_outputs = [self.experts[i](expert_inputs[i], expert_shared_inputs[i]) for i in range(self.num_experts)]
            out = dispatcher.combine(expert_outputs, multiply_by_gates=True)
        else:
            # Top-k experts for inference
            selected_experts = [self.experts[i] for i in top_k_indices.squeeze(0)]
            expert_outputs = torch.stack([expert(x, shared) for expert in selected_experts], dim=1)
            gates = gates.gather(1, top_k_indices)
            weighted_outputs = gates.unsqueeze(2).unsqueeze(3).unsqueeze(4) * expert_outputs
            out = weighted_outputs.sum(dim=1)
        out = self.proj_out(out)
        return out

class SparseDispatcher:
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), 
                            expert_out[-1].size(3), device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined


class DynamicPromptExpertModulator(nn.Module):
    def __init__(self, embed_dim=64, prompt_dim=96, prompt_len=5, prompt_size=96, num_blocks=3, align=False, mode_config=None):
        super(DynamicPromptExpertModulator, self).__init__()
        self.align = align
        self.prompt_dim = prompt_dim


        self.prompt_param = nn.Parameter(torch.rand(prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_proj = nn.Linear(embed_dim, prompt_len)
        self.conv = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)


        mode_config = mode_config.copy() if mode_config else {}

        if 'dim' in mode_config:
            del mode_config['dim']
        if 'freq_dim' in mode_config:  
            del mode_config['freq_dim']
            
        self.mode = AdapterLayer(dim=embed_dim, **mode_config)


    def forward(self, x):
        b, c, h, w = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_proj(emb), dim=1)
        input_conditioned_prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * \
                                  self.prompt_param.unsqueeze(0).repeat(b, 1, 1, 1, 1)
        input_conditioned_prompt = torch.sum(input_conditioned_prompt, dim=1)
        input_conditioned_prompt = F.interpolate(input_conditioned_prompt, (h, w), mode="bilinear")
        input_conditioned_prompt = self.conv(input_conditioned_prompt)


        output = self.mode(x, input_conditioned_prompt)

        return output



class PromptGuidedDeformableAlignment(ModulatedDeformConv2d):
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.prompt_dim = kwargs.pop('prompt_dim', 96)
        self.mode_config = kwargs.pop('mode_config', None)
        super(PromptGuidedDeformableAlignment, self).__init__(*args, **kwargs)

        self.proj = nn.Conv2d(2 * self.out_channels, self.out_channels, 3, 1, 1)
        self.DPEM_align = DynamicPromptExpertModulator(
            embed_dim=self.out_channels, prompt_dim=self.prompt_dim,
            num_blocks=1, align=True, mode_config=self.mode_config
        )
        self.fusion = nn.Conv2d(3 * self.out_channels, 2 * self.out_channels, 3, 1, 1, bias=True)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, cond, flow):
        proj_cond = self.proj(cond)
        integrated_features = self.DPEM_align(proj_cond)
        cond = self.fusion(torch.cat([integrated_features, cond], dim=1))
        cond = torch.cat([cond, flow], dim=1)
        out = self.conv_offset(cond)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                       self.dilation, self.groups, self.deform_groups)

@ARCH_REGISTRY.register()
class PGENet(nn.Module):
    def __init__(self,
                 mid_channels=96,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 spynet_pretrained=None,
                 keyframe_interval=6,
                 prompt_size=96,
                 prompt_dim=96,
                 cpu_cache_length=100,
                 mode_config=None):
        super().__init__()
        self.mid_channels = mid_channels
        self.prompt_dim = prompt_dim
        self.prompt_size = prompt_size
        self.cpu_cache_length = cpu_cache_length

        self.spynet = SpyNet(load_path=spynet_pretrained)
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, mid_channels, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlocksWithInputConv(mid_channels, mid_channels, 5))

        self.DPEM_prop = DynamicPromptExpertModulator(
            embed_dim=mid_channels, prompt_dim=self.prompt_dim,
            prompt_size=self.prompt_size, num_blocks=3, mode_config=mode_config
        )

        self.key_fusion = nn.Conv2d(2 * self.mid_channels, self.mid_channels, 3, 1, 1, bias=True)
        self.keyframe_interval = keyframe_interval

        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            self.deform_align[module] = PromptGuidedDeformableAlignment(
                mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=24,
                max_residue_magnitude=max_residue_magnitude,
                prompt_dim=self.prompt_dim,
                mode_config=mode_config
            )
            self.backbone[module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels, mid_channels, num_blocks)

        self.reconstruction = ResidualBlocksWithInputConv(5 * mid_channels, mid_channels, 5)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_keyframe_feature(self, x, keyframe_idx):
        feats_keyframe = {}
        for i in keyframe_idx:
            if self.cpu_cache:
                x_i = x[i].cuda()
            else:
                x_i = x[i]

            feats_keyframe[i] = self.DPEM_prop(x_i)

            if self.cpu_cache:
                feats_keyframe[i] = feats_keyframe[i].cpu()
                torch.cuda.empty_cache()
        return feats_keyframe

    def compute_flow(self, lqs):

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def upsample(self, lqs, feats):
        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)

            hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs):
        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU (no effect if using CPU)
        if t > self.cpu_cache_length and lqs.is_cuda:
            self.cpu_cache = True
        else:
            self.cpu_cache = False

        lqs_downsample = F.interpolate(
            lqs.view(-1, c, h, w), scale_factor=0.25,
            mode='bicubic').view(n, t, c, h//4, w//4)

        feats = {}
        # compute spatial features
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')

        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # generate keyframe features
        keyframe_idx = list(range(0, t, self.keyframe_interval))
        if keyframe_idx[-1] != t - 1:
            keyframe_idx.append(t - 1)  # last frame is a keyframe
        feats_keyframe = self.get_keyframe_feature(feats['spatial'], keyframe_idx)

        # feature propagation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                n, t, _, h, w = flows.size()
                frame_idx = list(range(0, t + 1))
                flow_idx = list(range(-1, t))
                mapping_idx = list(range(0, len(feats['spatial'])))
                mapping_idx += mapping_idx[::-1]

                if direction == 'backward':
                    frame_idx = frame_idx[::-1]
                    flow_idx = frame_idx

                feat_prop = flows.new_zeros(n, self.mid_channels, h, w)

                for i, idx in enumerate(frame_idx):
                    x_i = feats['spatial'][mapping_idx[idx]]
                    if self.cpu_cache:
                        x_i = x_i.cuda()
                        feat_prop = feat_prop.cuda()

                    pre_feat = feat_prop.clone()

                    if i > 0:
                        flow = flows[:, flow_idx[i], :, :, :]
                        if self.cpu_cache:
                            flow = flow.cuda()
                        feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

                    ## Prompt Guided Deformable Alignment
                    if i > 0:
                        cond = torch.cat([feat_prop, x_i], dim=1)
                        feat_prop = self.deform_align[module](pre_feat, cond, flow)
                        
                    
                    if idx in keyframe_idx:
                        if self.cpu_cache:
                            feats_keyframe_t = feats_keyframe[idx].cuda()
                        else:
                            feats_keyframe_t = feats_keyframe[idx]

                        feat_prop = self.key_fusion(torch.cat([feat_prop, feats_keyframe_t], dim=1))

                    # concatenate the residual info
                    feat = [x_i] + [
                        feats[k][idx]
                        for k in feats if k not in ['spatial', module]
                    ] + [feat_prop]
                    if self.cpu_cache:
                        feat = [f.cuda() for f in feat]

                    feat = torch.cat(feat, dim=1)
                    feat_prop = feat_prop + self.backbone[module](feat)
                    if self.cpu_cache:
                        feat_prop = feat_prop.cpu()
                        torch.cuda.empty_cache()
                    feats[module].append(feat_prop)

                if direction == 'backward':
                    feats[module] = feats[module][::-1]

                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)
