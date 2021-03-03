import torch
import torch.nn as nn
from torch.autograd import Function


class Quant8F(Function):

    @staticmethod
    def forward(cxt, input, dim):
        if not isinstance(dim, int):
            raise NotImplemented("Currently Only Support Selecting One Dimension.")
        #output = input.new(input.size())

        scale = (torch.max(input, dim=dim, keepdim=True)[0] - torch.min(input, dim=dim, keepdim=True)[0])


        initial_zero_point = 0 - torch.min(input, dim=dim, keepdim=True)[0] / scale
        # zero_point = 0

        initial_zero_point[initial_zero_point<0] = 0
        initial_zero_point[initial_zero_point>255] = 255


        # if initial_zero_point < 0:
        #     zero_point = 0
        # elif initial_zero_point > 255:
        #     zero_point = 255
        # else:
        #     zero_point = initial_zero_point

        #print("SCALE = {}".format(scale))
        #print("ZERO_POINT = {}".format(zero_point))

        #Replace with fake quantizatiaon

        #Reference: https://github.com/pytorch/pytorch/blob/master/torch/quantization/fake_quantize.py#L65
        output = ((input/scale + initial_zero_point).round_().clamp_(min=0, max=255) - initial_zero_point) * scale
        
        # dtype = torch.qint8
        # qm = nn.quantized.Quantize(scale, zero_point, dtype)
        # dqm = nn.quantized.DeQuantize()
        #
        # output = dqm(qm(input))
        
        #return output
        return output
        
    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# alias
quant8 = Quant8F.apply
