import torch
import torch.nn as nn
from torch.autograd import Function
import math

class Quant8F(Function):

    @staticmethod
    def forward(cxt, input):
        #output = input.new(input.size())

        scale = (torch.max(input) - torch.min(input)) / 255

        initial_zero_point = 0 - torch.min(input) / scale
        zero_point = 0
        if initial_zero_point < 0:
            zero_point = 0
        elif initial_zero_point > 255*scale:
            zero_point = 255*scale
        else:
            zero_point = initial_zero_point
            if math.isnan(zero_point):
                zero_point = 0
        zero_point = int(zero_point)
        #print("SCALE = {}".format(scale))
        #print("ZERO_POINT = {}".format(zero_point))
        
        dtype = torch.qint8
        qm = nn.quantized.Quantize(scale, zero_point, dtype)
        dqm = nn.quantized.DeQuantize()

        output = dqm(qm(input))        

        #mse_loss = nn.MSELoss()
        #loss = mse_loss(input, output)
        #print("Quantization loss: {}".format(loss))
        
        return output
        
    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# alias
quant8 = Quant8F.apply
