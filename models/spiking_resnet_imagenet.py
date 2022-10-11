import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function

__all__ = ['OnlineSpikingNFResNet', 'online_spiking_nfresnet18', 'online_spiking_nfresnet34', 'online_spiking_nfresnet50', 'online_spiking_nfresnet101',
           'online_spiking_nfresnet152', 'online_spiking_nfresnext50_32x4d', 'online_spiking_nfresnext101_32x8d',
           'online_spiking_nfwide_resnet50_2', 'online_spiking_nfwide_resnet101_2']

# modified by https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

class ScaledWSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, gain=True, eps=1e-4):
        super(ScaledWSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        else:
            self.gain = None
        self.eps = eps

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
        weight = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


def wsconv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return ScaledWSConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=True, gain=True, dilation=dilation)


def wsconv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return ScaledWSConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True, gain=True)


class Replace(Function):
    @staticmethod
    def forward(ctx, z1, z1_r):
        return z1_r

    @staticmethod
    def backward(ctx, grad):
        return (grad, grad)


class WrapedSNNOp(nn.Module):

    def __init__(self, op):
        super(WrapedSNNOp, self).__init__()
        self.op = op

    def forward(self, x, **kwargs):
        require_wrap = kwargs.get('require_wrap', True)
        if require_wrap:
            B = x.shape[0] // 2
            spike = x[:B]
            rate = x[B:]
            with torch.no_grad():
                out = self.op(spike).detach()
            in_for_grad = Replace.apply(spike, rate)
            out_for_grad = self.op(in_for_grad)
            output = Replace.apply(out_for_grad, out)
            return output
        else:
            return self.op(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, weight_standardization=True, beta=1.0, alpha=1.0,
                 single_step_neuron: callable = None, stochdepth_rate=0.0, **kwargs):
        super(BasicBlock, self).__init__()
        self.stochdepth_rate = stochdepth_rate
        self.grad_with_rate = kwargs.get('grad_with_rate', False)

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if weight_standardization:
            self.conv1 = wsconv3x3(inplanes, planes, stride)
            self.conv2 = wsconv3x3(planes, planes)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes)
        self.sn1 = single_step_neuron(**kwargs)
        self.sn2 = single_step_neuron(**kwargs)
        self.downsample = downsample
        self.stride = stride

        self.beta, self.alpha = beta, alpha
        self.skipinit_gain = nn.Parameter(torch.zeros(()))

        if self.grad_with_rate:
            self.conv1 = WrapedSNNOp(self.conv1)
            self.conv2 = WrapedSNNOp(self.conv2)
            if self.downsample != None:
                self.downsample = WrapedSNNOp(self.downsample)

    def forward(self, x, **kwargs):
        require_wrap = self.grad_with_rate and self.training
        out = x / self.beta

        if require_wrap:
            out = self.sn1(out, output_type='spike_rate', **kwargs)
        else:
            out = self.sn1(out, **kwargs)
        out = out * 2.74

        if self.downsample is not None:
            if require_wrap:
                identity = self.downsample(out, require_wrap=True)
            else:
                if self.grad_with_rate:
                    identity = self.downsample(out, require_wrap=False)
                else:
                    identity = self.downsample(out)
        else:
            identity = x

        # stochastic depth
        # TODO consider the multi-gpu condition?
        if self.stochdepth_rate > 0.0 and self.training:
            init = kwargs.get('init', False)
            if init:
                self.stochdepth = torch.empty([1])
                self.stochdepth = self.stochdepth.bernoulli_(1 - self.stochdepth_rate)
            if torch.equal(self.stochdepth, torch.zeros(1)):
                return identity

        if require_wrap:
            out = self.conv1(out, require_wrap=True)
            out = self.sn2(out, output_type='spike_rate', **kwargs)
            out = out * 2.74

            out = self.conv2(out, require_wrap=True)
        else:
            if self.grad_with_rate:
                out = self.conv1(out, require_wrap=False)
            else:
                out = self.conv1(out)
            out = self.sn2(out, **kwargs)
            out = out * 2.74

            if self.grad_with_rate:
                out = self.conv2(out, require_wrap=False)
            else:
                out = self.conv2(out)

        out = out * self.skipinit_gain * self.alpha + identity

        return out

    def get_spike(self):
        spikes = []
        spike = self.sn1.spike.cpu()
        spikes.append(spike.reshape(spike.shape[0], -1))
        spike = self.sn2.spike.cpu()
        spikes.append(spike.reshape(spike.shape[0], -1))
        return spikes


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, weight_standardization=True, beta=1.0, alpha=1.0, 
                 single_step_neuron: callable = None, stochdepth_rate=0.0, **kwargs):
        super(Bottleneck, self).__init__()
        self.stochdepth_rate = stochdepth_rate
        self.grad_with_rate = kwargs.get('grad_with_rate', False)

        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        if weight_standardization:
            self.conv1 = wsconv1x1(inplanes, width)
            self.conv2 = wsconv3x3(width, width, stride, groups, dilation)
            self.conv3 = wsconv1x1(width, planes * self.expansion)
        else:
            self.conv1 = conv1x1(inplanes, width)
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
            self.conv3 = conv1x1(width, planes * self.expansion)

        self.sn1 = single_step_neuron(**kwargs)
        self.sn2 = single_step_neuron(**kwargs)
        self.sn3 = single_step_neuron(**kwargs)
        self.downsample = downsample
        self.stride = stride

        self.beta, self.alpha = beta, alpha
        self.skipinit_gain = nn.Parameter(torch.zeros(()))

        if self.grad_with_rate:
            self.conv1 = WrapedSNNOp(self.conv1)
            self.conv2 = WrapedSNNOp(self.conv2)
            self.conv3 = WrapedSNNOp(self.conv3)
            if self.downsample != None:
                self.downsample = WrapedSNNOp(self.downsample)

    def forward(self, x, **kwargs):
        require_wrap = self.grad_with_rate and self.training
        out = x / self.beta

        if require_wrap:
            out = self.sn1(out, output_type='spike_rate', **kwargs)
        else:
            out = self.sn1(out, **kwargs)
        out = out * 2.74

        if self.downsample is not None:
            if require_wrap:
                identity = self.downsample(out, require_wrap=True)
            else:
                if self.grad_with_rate:
                    identity = self.downsample(out, require_wrap=False)
                else:
                    identity = self.downsample(out)
        else:
            identity = x

        # stochastic depth
        # TODO consider the multi-gpu condition?
        if self.stochdepth_rate > 0.0 and self.training:
            init = kwargs.get('init', False)
            if init:
                self.stochdepth = torch.empty([1])
                self.stochdepth = self.stochdepth.bernoulli_(1 - self.stochdepth_rate)
            if torch.equal(self.stochdepth, torch.zeros(1)):
                return identity

        if require_wrap:
            out = self.conv1(out, require_wrap=True)
            out = self.sn2(out, output_type='spike_rate', **kwargs)
            out = out * 2.74
            out = self.conv2(out, require_wrap=True)
            out = self.sn3(out, output_type='spike_rate', **kwargs)
            out = out * 2.74
            out = self.conv3(out, require_wrap=True)
        else:
            if self.grad_with_rate:
                out = self.conv1(out, require_wrap=False)
            else:
                out = self.conv1(out)
            out = self.sn2(out, output_type='spike_rate', **kwargs)
            out = out * 2.74
            if self.grad_with_rate:
                out = self.conv2(out, require_wrap=False)
            else:
                out = self.conv2(out)
            out = self.sn3(out, output_type='spike_rate', **kwargs)
            out = out * 2.74
            if self.grad_with_rate:
                out = self.conv3(out, require_wrap=False)
            else:
                out = self.conv3(out)

        out = out * self.skipinit_gain * self.alpha + identity

        return out

    def get_spike(self):
        spikes = []
        spike = self.sn1.spike.cpu()
        spikes.append(spike.reshape(spike.shape[0], -1))
        spike = self.sn2.spike.cpu()
        spikes.append(spike.reshape(spike.shape[0], -1))
        spike = self.sn3.spike.cpu()
        spikes.append(spike.reshape(spike.shape[0], -1))
        return spikes


class SequentialModule(nn.Sequential):
    def forward(self, input, **kwargs):
        for module in self._modules.values():
            input = module(input, **kwargs)
        return input

    def get_spike(self):
        spikes = []
        for module in self._modules.values():
            spikes_module = module.get_spike()
            spikes += spikes_module
        return spikes


class OnlineSpikingNFResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, 
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 weight_standardization=True, single_step_neuron: callable = None,
                 alpha=0.2, drop_rate=0.0, **kwargs):
        super(OnlineSpikingNFResNet, self).__init__()
        self.ws = weight_standardization
        self.alpha = alpha
        self.drop_rate = drop_rate

        self.inplanes = 64
        self.dilation = 1
        self.c_in = kwargs.get('c_in', 3)
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if weight_standardization:
            self.conv1 = ScaledWSConv2d(self.c_in, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True, gain=True)
        else:
            self.conv1 = nn.Conv2d(self.c_in, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        expected_var = 1.0
        self.layer1, expected_var = self._make_layer(block, 64, layers[0], alpha=self.alpha, var=expected_var, single_step_neuron=single_step_neuron, **kwargs)
        self.layer2, expected_var = self._make_layer(block, 128, layers[1], stride=2, alpha=self.alpha, var=expected_var,
                                                     dilate=replace_stride_with_dilation[0], single_step_neuron=single_step_neuron, **kwargs)
        self.layer3, expected_var = self._make_layer(block, 256, layers[2], stride=2, alpha=self.alpha, var=expected_var,
                                                     dilate=replace_stride_with_dilation[1], single_step_neuron=single_step_neuron, **kwargs)
        self.layer4, expected_var = self._make_layer(block, 512, layers[3], stride=2, alpha=self.alpha, var=expected_var,
                                                     dilate=replace_stride_with_dilation[2], single_step_neuron=single_step_neuron, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #if self.drop_rate > 0.0:
        #    self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        torch.nn.init.zeros_(self.fc.weight)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, ScaledWSConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, alpha=1.0, var=1.0, dilate=False, single_step_neuron: callable = None, **kwargs):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.ws:
                downsample = wsconv1x1(self.inplanes, planes * block.expansion, stride)
            else:
                downsample = conv1x1(self.inplanes, planes * block.expansion, stride)

        layers = []
        beta = var ** 0.5
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, self.ws, beta, alpha, single_step_neuron, **kwargs))
        self.inplanes = planes * block.expansion
        if downsample != None:
            var = 1. + self.alpha ** 2
        else:
            var += self.alpha ** 2
        for _ in range(1, blocks):
            beta = var ** 0.5
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                weight_standardization=self.ws, beta=beta, alpha=alpha, single_step_neuron=single_step_neuron, **kwargs))
            var += self.alpha ** 2

        return SequentialModule(*layers), var

    def _forward_impl(self, x, **kwargs):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x, **kwargs)
        x = self.layer2(x, **kwargs)
        x = self.layer3(x, **kwargs)
        x = self.layer4(x, **kwargs)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # TODO consider the multi gpu condition?
        if self.drop_rate > 0.0 and self.training:
            #x = self.dropout(x)
            init = kwargs.get('init', False)
            if init:
                self.mask = torch.zeros_like(x).bernoulli_(1 - self.drop_rate)
                self.mask = self.mask.requires_grad_(False) / (1 - self.drop_rate)
            x = self.mask.expand_as(x) * x
        x = self.fc(x)

        return x

    def forward(self, x, **kwargs):
        return self._forward_impl(x, **kwargs)

    def get_spike(self):
        spikes = []
        spikes += self.layer1.get_spike()
        spikes += self.layer2.get_spike()
        spikes += self.layer3.get_spike()
        spikes += self.layer4.get_spike()
        return spikes


def _online_spiking_resnet(arch, block, layers, pretrained, progress, single_step_neuron, **kwargs):
    model = OnlineSpikingNFResNet(block, layers, single_step_neuron=single_step_neuron, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def online_spiking_nfresnet18(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    return _online_spiking_resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, single_step_neuron, **kwargs)

def online_spiking_nfresnet34(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    return _online_spiking_resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, single_step_neuron, **kwargs)

def online_spiking_nfresnet50(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    return _online_spiking_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, single_step_neuron, **kwargs)

def online_spiking_nfresnet101(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    return _online_spiking_resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, single_step_neuron, **kwargs)

def online_spiking_nfresnet152(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    return _online_spiking_resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, single_step_neuron, **kwargs)

def online_spiking_nfresnext50_32x4d(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _online_spiking_resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, single_step_neuron, **kwargs)

def online_spiking_nfresnext101_32x8d(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _online_spiking_resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, single_step_neuron, **kwargs)

def online_spiking_nfwide_resnet50_2(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _online_spiking_resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, single_step_neuron, **kwargs)

def online_spiking_nfwide_resnet101_2(pretrained=False, progress=True, single_step_neuron: callable=None, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _online_spiking_resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, progress, single_step_neuron, **kwargs)

