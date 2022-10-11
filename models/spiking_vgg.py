import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function

__all__ = [
    'OnlineSpikingVGG', 'online_spiking_vgg11', 'online_spiking_vgg11_ws', 'online_spiking_vgg11f_ws',
]

# modified by https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py

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


class ScaledWSLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, gain=True, eps=1e-4):
        super(ScaledWSLinear, self).__init__(in_features, out_features, bias)
        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_features, 1))
        else:
            self.gain = None
        self.eps = eps

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1], keepdims=True)
        var = torch.var(self.weight, axis=[1], keepdims=True)
        weight = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x):
        return F.linear(x, self.get_weight(), self.bias)


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


class SequentialModule(nn.Sequential):

    def __init__(self, single_step_neuron, *args):
        super(SequentialModule, self).__init__(*args)
        self.single_step_neuron = single_step_neuron

    def forward(self, input, **kwargs):
        for module in self._modules.values():
            if isinstance(module, self.single_step_neuron) or isinstance(module, WrapedSNNOp):
                input = module(input, **kwargs)
            else:
                input = module(input)
        return input

    def get_spike(self):
        spikes = []
        for module in self._modules.values():
            if isinstance(module, self.single_step_neuron):
                spike = module.spike.cpu()
                spikes.append(spike.reshape(spike.shape[0], -1))
        return spikes


class Scale(nn.Module):

    def __init__(self, scale):
        super(Scale, self).__init__()
        self.scale = scale

    def forward(self, x, **kwargs):
        return x * self.scale


class OnlineSpikingVGG(nn.Module):

    def __init__(self, cfg, weight_standardization=True, num_classes=1000, init_weights=True,
                 single_step_neuron: callable = None, light_classifier=True, BN=False, **kwargs):
        super(OnlineSpikingVGG, self).__init__()
        self.single_step_neuron = single_step_neuron
        self.grad_with_rate = kwargs.get('grad_with_rate', False)
        self.fc_hw = kwargs.get('fc_hw', 3)
        self.features = self.make_layers(cfg=cfg, weight_standardization=weight_standardization,
                                         neuron=single_step_neuron, BN=BN, **kwargs)
        if light_classifier:
            self.avgpool = nn.AdaptiveAvgPool2d((self.fc_hw, self.fc_hw))
            if self.grad_with_rate:
                self.classifier = SequentialModule(
                    single_step_neuron, # not in the module, but parameter
                    WrapedSNNOp(nn.Linear(512*(self.fc_hw**2), num_classes)),
                )
            else:
                self.classifier = SequentialModule(
                    single_step_neuron, # not in the module, but parameter
                    nn.Linear(512*(self.fc_hw**2), num_classes),
                )
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            if self.grad_with_rate:
                self.classifier = SequentialModule(
                    single_step_neuron,
                    WrapedSNNOp(ScaledWSLinear(512 * 7 * 7, 4096)),
                    single_step_neuron(**kwargs, neuron_dropout=0.0),
                    Scale(2.74),
                    nn.Dropout(),
                    WrapedSNNOp(ScaledWSLinear(4096, 4096)),
                    single_step_neuron(**kwargs, neuron_dropout=0.0),
                    Scale(2.74),
                    nn.Dropout(),
                    WrapedSNNOp(nn.Linear(4096, num_classes)),
                )
            else:
                self.classifier = SequentialModule(
                    single_step_neuron,
                    ScaledWSLinear(512 * 7 * 7, 4096),
                    single_step_neuron(**kwargs, neuron_dropout=0.0),
                    Scale(2.74),
                    nn.Dropout(),
                    ScaledWSLinear(4096, 4096),
                    single_step_neuron(**kwargs, neuron_dropout=0.0),
                    Scale(2.74),
                    nn.Dropout(),
                    nn.Linear(4096, num_classes),
                )
        if init_weights:
            self._initialize_weights()

    def forward(self, x, **kwargs):
        require_wrap = self.grad_with_rate and self.training
        if require_wrap:
            x = self.features(x, output_type='spike_rate', require_wrap=True, **kwargs)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x, output_type='spike_rate', require_wrap=True, **kwargs)
        else:
            x = self.features(x, require_wrap=False, **kwargs)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x, require_wrap=False, **kwargs)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, ScaledWSConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(cfg, weight_standardization=True, neuron: callable = None, BN=False, **kwargs):
        grad_with_rate = kwargs.get('grad_with_rate', False)
        layers = []
        in_channels = kwargs.get('c_in', 3)
        first_conv = True
        use_stride_2 = False
        for v in cfg:
            if v == 'M':
                #layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif v == 'S':
                use_stride_2 = True
            else:
                if use_stride_2:
                    stride = 2
                    use_stride_2 = False
                else:
                    stride = 1
                if weight_standardization:
                    if first_conv:
                        conv2d = ScaledWSConv2d(in_channels, v, kernel_size=3, padding=1, stride=stride)
                        first_conv = False
                    else:
                        if grad_with_rate:
                            conv2d = WrapedSNNOp(ScaledWSConv2d(in_channels, v, kernel_size=3, padding=1, stride=stride))
                        else:
                            conv2d = ScaledWSConv2d(in_channels, v, kernel_size=3, padding=1, stride=stride)
                    layers += [conv2d, neuron(**kwargs), Scale(2.74)]
                else:
                    if first_conv:
                        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride)
                        first_conv = False
                    else:
                        if grad_with_rate:
                            conv2d = WrapedSNNOp(nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride))
                        else:
                            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride)
                    if BN:
                        bn = nn.BatchNorm2d(v)
                        layers += [conv2d, bn, neuron(**kwargs)]
                    else:
                        layers += [conv2d, neuron(**kwargs), Scale(2.74)]
                in_channels = v
        return SequentialModule(neuron, *layers)

    def get_spike(self):
        return self.features.get_spike()



cfgs = {
    'A': [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],#, 'M'],
}


def _spiking_vgg(arch, cfg, weight_standardization, pretrained, progress, single_step_neuron: callable = None, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = OnlineSpikingVGG(cfg=cfgs[cfg], weight_standardization=weight_standardization, single_step_neuron=single_step_neuron, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def online_spiking_vgg11(pretrained=False, progress=True, single_step_neuron: callable = None, **kwargs):
    return _spiking_vgg('vgg11', 'A', False, pretrained, progress, single_step_neuron, **kwargs)


def online_spiking_vgg11_ws(pretrained=False, progress=True, single_step_neuron: callable = None, **kwargs):
    return _spiking_vgg('vgg11', 'A', True, pretrained, progress, single_step_neuron, **kwargs)



class OnlineSpikingVGGF(nn.Module):

    def __init__(self, cfg, weight_standardization=True, num_classes=1000, init_weights=True,
                 single_step_neuron: callable = None, light_classifier=True, BN=False, **kwargs):
        super(OnlineSpikingVGGF, self).__init__()
        self.single_step_neuron = single_step_neuron
        self.grad_with_rate = kwargs.get('grad_with_rate', False)
        self.fc_hw = kwargs.get('fc_hw', 3)
        self.conv1, self.features = self.make_layers(cfg=cfg, weight_standardization=weight_standardization,
                                         neuron=single_step_neuron, BN=BN, **kwargs)

        # feedback connections
        scale_factor = 1
        for v in cfg:
            if v == 'M' or v == 'S':
                scale_factor *= 2
        self.up = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        # TODO consider general cfg settings
        self.fb_conv = nn.Conv2d(cfg[-1], cfg[0], kernel_size=3, padding=1, stride=1)
        if self.grad_with_rate:
            self.fb_conv = WrapedSNNOp(self.fb_conv)


        if light_classifier:
            self.avgpool = nn.AdaptiveAvgPool2d((self.fc_hw, self.fc_hw))
            if self.grad_with_rate:
                self.classifier = SequentialModule(
                    single_step_neuron, # not in the module, but parameter
                    WrapedSNNOp(nn.Linear(512*(self.fc_hw**2), num_classes)),
                )
            else:
                self.classifier = SequentialModule(
                    single_step_neuron, # not in the module, but parameter
                    nn.Linear(512*(self.fc_hw**2), num_classes),
                )
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            if self.grad_with_rate:
                self.classifier = SequentialModule(
                    single_step_neuron,
                    WrapedSNNOp(ScaledWSLinear(512 * 7 * 7, 4096)),
                    single_step_neuron(**kwargs, neuron_dropout=0.0),
                    Scale(2.74),
                    nn.Dropout(),
                    WrapedSNNOp(ScaledWSLinear(4096, 4096)),
                    single_step_neuron(**kwargs, neuron_dropout=0.0),
                    Scale(2.74),
                    nn.Dropout(),
                    WrapedSNNOp(nn.Linear(4096, num_classes)),
                )
            else:
                self.classifier = SequentialModule(
                    single_step_neuron,
                    ScaledWSLinear(512 * 7 * 7, 4096),
                    single_step_neuron(**kwargs, neuron_dropout=0.0),
                    Scale(2.74),
                    nn.Dropout(),
                    ScaledWSLinear(4096, 4096),
                    single_step_neuron(**kwargs, neuron_dropout=0.0),
                    Scale(2.74),
                    nn.Dropout(),
                    nn.Linear(4096, num_classes),
                )
        if init_weights:
            self._initialize_weights()

    def forward(self, x, **kwargs):
        require_wrap = self.grad_with_rate and self.training
        init = kwargs.get('init', False)
        if require_wrap:
            if init:
                x = self.conv1(x)
            else:
                x = self.conv1(x) + self.fb_conv(self.up(self.fb_features), require_wrap=True)
            x = self.features(x, output_type='spike_rate', require_wrap=True, **kwargs)

            self.fb_features = x.clone().detach()

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x, output_type='spike_rate', require_wrap=True, **kwargs)
        else:
            if init:
                x = self.conv1(x)
            else:
                if self.grad_with_rate:
                    x = self.conv1(x) + self.fb_conv(self.up(self.fb_features), require_wrap=False)
                else:
                    x = self.conv1(x) + self.fb_conv(self.up(self.fb_features))
            x = self.features(x, require_wrap=False, **kwargs)

            self.fb_features = x.clone().detach()

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x, require_wrap=False, **kwargs)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, ScaledWSConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
        # zero initialization for fb conv
        if self.grad_with_rate:
            nn.init.constant_(self.fb_conv.op.weight, 0.)
        else:
            nn.init.constant_(self.fb_conv.weight, 0.)

    @staticmethod
    def make_layers(cfg, weight_standardization=True, neuron: callable = None, BN=False, **kwargs):
        grad_with_rate = kwargs.get('grad_with_rate', False)
        layers = []
        in_channels = kwargs.get('c_in', 3)
        first_conv = True
        use_stride_2 = False
        for v in cfg:
            if v == 'M':
                #layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif v == 'S':
                use_stride_2 = True
            else:
                if use_stride_2:
                    stride = 2
                    use_stride_2 = False
                else:
                    stride = 1
                if weight_standardization:
                    if first_conv:
                        conv1 = ScaledWSConv2d(in_channels, v, kernel_size=3, padding=1, stride=stride)
                        first_conv = False
                        layers += [neuron(**kwargs), Scale(2.74)]
                    else:
                        if grad_with_rate:
                            conv2d = WrapedSNNOp(ScaledWSConv2d(in_channels, v, kernel_size=3, padding=1, stride=stride))
                        else:
                            conv2d = ScaledWSConv2d(in_channels, v, kernel_size=3, padding=1, stride=stride)
                        layers += [conv2d, neuron(**kwargs), Scale(2.74)]
                else:
                    if first_conv:
                        conv1 = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride)
                        first_conv = False
                        if BN:
                            bn = nn.BatchNorm2d(v)
                            layers += [bn, neuron(**kwargs)]
                        else:
                            layers += [neuron(**kwargs), Scale(2.74)]
                    else:
                        if grad_with_rate:
                            conv2d = WrapedSNNOp(nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride))
                        else:
                            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride)
                        if BN:
                            bn = nn.BatchNorm2d(v)
                            layers += [conv2d, bn, neuron(**kwargs)]
                        else:
                            layers += [conv2d, neuron(**kwargs), Scale(2.74)]
                in_channels = v
        return conv1, SequentialModule(neuron, *layers)

    def get_spike(self):
        return self.features.get_spike()


def _spiking_vggf(arch, cfg, weight_standardization, pretrained, progress, single_step_neuron: callable = None, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = OnlineSpikingVGGF(cfg=cfgs[cfg], weight_standardization=weight_standardization, single_step_neuron=single_step_neuron, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def online_spiking_vgg11f_ws(pretrained=False, progress=True, single_step_neuron: callable = None, **kwargs):
    return _spiking_vggf('vgg11', 'A', True, pretrained, progress, single_step_neuron, **kwargs)
