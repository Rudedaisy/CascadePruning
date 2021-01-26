import torch
import torch.nn as nn
from torch.autograd import Function


class Quant8F(Function):

    @staticmethod
    def forward(cxt, input):
        #output = input.new(input.size())

        scale = (torch.max(input) - torch.min(input))

        initial_zero_point = 0 - torch.min(input) / scale
        zero_point = 0
        if initial_zero_point < 0:
            zero_point = 0
        elif initial_zero_point > 255:
            zero_point = 255
        else:
            zero_point = initial_zero_point
        zero_point = int(zero_point)
        #print("SCALE = {}".format(scale))
        #print("ZERO_POINT = {}".format(zero_point))
        
        dtype = torch.qint8
        qm = nn.quantized.Quantize(scale, zero_point, dtype)
        dqm = nn.quantized.DeQuantize()

        output = dqm(qm(input))        
        
        #return output
        return input
        
    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# alias
quant8 = Quant8F.apply
