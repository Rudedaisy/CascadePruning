import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MyConv2d(nn.Module):
    def __init__(self, n_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1):
        super(MyConv2d, self).__init__()

        self.kernel_size = (kernel_size, kernel_size)
        self.kernal_size_number = kernel_size * kernel_size
        #self.kernel_single_size = kernel_size
        self.out_channels = out_channels
        self.dilation = (dilation, dilation)
        self.padding = (padding, padding)
        self.stride = (stride, stride)
        self.n_channels = n_channels
        #self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.n_channels, self.kernal_size_number))
        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.n_channels, kernel_size, kernel_size))

    def forward(self, x):
        width = self.calculateNewWidth(x)
        height = self.calculateNewHeight(x)
        windows = self.calculateWindows(x).to(device)
        
        result = torch.zeros(
            [x.shape[0] * self.out_channels, width, height], dtype=torch.float32, device=device
        )
        print("FORWARD")
        for channel in range(x.shape[1]):
            for i_convNumber in range(self.out_channels):
                #print(" ---- shapes ----")
                #print(windows[channel].shape)
                #print(self.weight[i_convNumber][channel].shape)
                xx = torch.matmul(windows[channel], self.weight[i_convNumber][channel].reshape(-1)) 
                xx = xx.view(-1, width, height).to(device)
                result[i_convNumber * xx.shape[0] : (i_convNumber + 1) * xx.shape[0]] += xx # goofy indexing is for additional dimension (batch size)
                
        result = result.view(x.shape[0], self.out_channels, width, height)
        return result  

    def calculateWindows(self, x):
        windows = F.unfold(
            x, kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation, stride=self.stride
        )

        windows = windows.transpose(1, 2).contiguous().view(-1, x.shape[1], self.kernal_size_number)
        windows = windows.transpose(0, 1)

        return windows

    def calculateNewWidth(self, x):
        return (
            (x.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
            // self.stride[0]
        ) + 1

    def calculateNewHeight(self, x):
        return (
            (x.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)
            // self.stride[1]
        ) + 1

def main():

    conv = MyConv2d(3, 1, 3).to(device)
    x = torch.randn(1, 3, 24, 24).to(device)
    out = conv(x)
    out.mean().backward()
    print(conv.weight.grad)

if __name__ == "__main__":
    main()
