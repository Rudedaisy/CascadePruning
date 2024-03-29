import torch
from quantize import Quant8F

class casConv2d(torch.nn.Module):
    # def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
    #              dilation=1, device=None, bias=True, quant_func = Quant8F, array_size=32):
    #     super(casConv2d, self).__init__()
    #
    #     if isinstance(stride, int):
    #         stride = (stride, stride)
    #     if isinstance(padding, int):
    #         padding = (padding, padding)
    #     if isinstance(dilation, int):
    #         dilation = (dilation, dilation)
    #
    #     self.in_channels = in_channels
    #     self.out_channels = out_channels
    #     self.kernel_size = kernel_size
    #     self.stride = stride
    #     self.padding = padding
    #     self.dilation = dilation
    #     self.quant_func = quant_func
    #     self.device = device
    #     self.array_size=array_size
    #     if bias:
    #         self.bias = torch.nn.Parameter(torch.randn((out_channels, )), requires_grad=True)
    #     else:
    #         self.bias = None
    #     self.weight = torch.nn.Parameter(torch.randn((out_channels, in_channels, kernel_size[0], kernel_size[1] )),
    #                                      requires_grad=True)


    def __init__(self, conv2d_module, array_size=32, quant=8, quant_func = Quant8F):
        super(casConv2d, self).__init__()

        #assert isinstance(conv2d_module, torch.nn.Conv2d), "Input Module is not a valid conv operator!"
        self.conv2d_module = conv2d_module
        self.in_channels = conv2d_module.in_channels
        self.out_channels = conv2d_module.out_channels
        self.kernel_size = conv2d_module.kernel_size
        self.stride = conv2d_module.stride
        self.padding = conv2d_module.padding
        self.dilation = conv2d_module.dilation
        self.quant_func = quant_func()
        self.quant = quant
        #self.device = conv2d_module.device
        self.bias = conv2d_module.bias
        self.array_size = array_size
        self.weight = conv2d_module.weight

        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        if isinstance(self.padding, int):
            self.padding = (self.padding, self.padding)
        if isinstance(self.dilation, int):
            self.dilation = (self.dilation, self.dilation)

    def set_quant_params(self, x, quant=8):
        dim = 1
        out = self.conv2d_module.forward(x)
        out = out.reshape(1, self.out_channels, -1)

        scale = (1.0/(2**quant-1)) * (torch.max(out, dim=dim, keepdim=False)[0] - torch.min(out, dim=dim, keepdim=False)[0])
        #assert torch.sum(scale == 0).data[0] == 0
        initial_zero_point = 0 + -1*torch.div(torch.min(out, dim=dim, keepdim=False)[0], scale)
        initial_zero_point[initial_zero_point<0] = 0
        initial_zero_point[initial_zero_point>(2**quant-1)] = (2**quant-1)
        initial_zero_point[initial_zero_point != initial_zero_point] = 0
        initial_zero_point = initial_zero_point.int()

        scale = scale.repeat(1, 1, x.shape[2], x.shape[3])
        initial_zero_point = initial_zero_point.repeat(1, 1, x.shape[2], x.shape[3])

        self.scale = scale
        self.initial_zero_point = initial_zero_point
        return

    def forward(self, x):
        #Standard conv2d to see mse result
        #output_baseline = self.conv2d_module.forward(x)
        self.set_quant_params(x, self.quant)
        
        #Calculate Output shape
        o_shape = [(x.shape[2] + 2*self.padding[0]-self.dilation[0] * (self.kernel_size[0]-1) - 1)//self.stride[0] + 1,
                   (x.shape[3] + 2*self.padding[1] -self.dilation[1] * (self.kernel_size[1]-1) - 1)//self.stride[1] + 1]

        #x has the shape of [batch_size, channel, w, h]
        candi_blocks = torch.nn.functional.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride)

        #Now candi_block has the shape of [batch_size, 1, in_channels * k * k, w_out * h_out ]
        candi_blocks = candi_blocks.reshape(x.shape[0], 1, x.shape[1] * self.kernel_size[0]*self.kernel_size[1],
                                            candi_blocks.shape[2])

        #Now we preprocess the weight matrix to align with candi_blocks
        #Weight has the shape of [out_channels, in_channels, k , k]
        candi_weights = self.weight.reshape(1, self.out_channels, self.in_channels * self.kernel_size[0] * self.kernel_size[1], 1)

        #Weight has the shape of [1, out_channels, in_channels * k * k, 1]

        #Match the dimension with broadcasting
        pre_quant_result = candi_blocks * candi_weights
        del candi_blocks
        del candi_weights


        #Pre_quant_result now have the shape of [batch_size, out_channels, in_channels * k * k, w_out * h_out ]
        #Now we need to pad the result to be the multiple of array size
        #Calculate the padding needed for each dimension
        pad_d3 = self.array_size - pre_quant_result.shape[2] % self.array_size
        pre_quant_result = torch.nn.functional.pad(pre_quant_result, (0, 0, pad_d3, 0), mode='constant', value=0)


        if self.array_size > 1:
            #Now we split it to chunks and calculate the sum
            pre_quant_split_result = pre_quant_result.reshape(pre_quant_result.shape[0], pre_quant_result.shape[1],
                                                              pre_quant_result.shape[2]//self.array_size, self.array_size, pre_quant_result.shape[-1])
            del pre_quant_result
            pre_quant_accu = torch.sum(pre_quant_split_result, dim=3)
            del pre_quant_split_result
        else:
            pre_quant_accu = pre_quant_result
            del pre_quant_result
        #Now we quantize the tensor along partial sum dimension
        #for chunk in range(pre_quant_accu.shape[2]):
        #    pre_quant_accu[:,:,chunk] = self.quant_func.apply(pre_quant_accu[:,:,chunk])
        quant_accu = self.quant_func.apply(pre_quant_accu, 2, self.quant, self.scale, self.initial_zero_point)
        del pre_quant_accu
        #quant_accu = pre_quant_accu
        
        #Let's accumulate these and get the result
        output = torch.sum(quant_accu, dim=2).reshape(quant_accu.shape[0], quant_accu.shape[1], o_shape[0], o_shape[1])

        #output = self.quant_func.apply(output, 1)
        
        #loss_func = torch.nn.MSELoss()
        #loss_result = loss_func(output_baseline, output)
        #print("Layer loss: {}".format(loss_result))
        
        return output












