import torch
from torch import Tensor
from torch.nn import Module, Conv1d, BatchNorm1d, ReLU

class DilatedConvBLock1D(Module):
    def __init__(self, in_channels, out_channels, input_size, kernel_size, stride, dilation, groups=1):
        super().__init__()
        assert groups == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.input_size = input_size
        padding = int((input_size * (self.stride - 1) - self.stride + self.dilation * (self.kernel_size - 1) + 1) / 2)
        # Conv - Norm - ReLU - Conv - Norm
        self.cnn1 = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True, groups=groups)
        self.batch_norm1 = BatchNorm1d(num_features=out_channels)
        self.non_linearity = ReLU(inplace=True)
        self.cnn2 = Conv1d( in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True, groups=groups)
        self.batch_norm2 = BatchNorm1d(num_features=out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.batch_norm1(self.non_linearity(self.cnn1(x)))
        x = self.batch_norm2(self.non_linearity(self.cnn2(x))) + x
        return x

class DilatedConv(Module):
    def __init__(self, in_channels, dilation_rate, input_size, kernel_size, stride):
        super().__init__()
        self.blks = torch.nn.ModuleList()
        self.blks.append(DilatedConvBLock1D(in_channels, in_channels // 2, input_size=input_size, kernel_size=kernel_size, stride=stride, dilation=1))
        self.blks.append(DilatedConvBLock1D(in_channels // 2, in_channels // 4, input_size=input_size, kernel_size=kernel_size, stride=stride, dilation=dilation_rate))
        self.blks.append(DilatedConvBLock1D(in_channels // 4, in_channels // 8, input_size=input_size, kernel_size=kernel_size, stride=stride, dilation=dilation_rate))
        self.blks.append(DilatedConvBLock1D(in_channels // 8, in_channels // 16, input_size=input_size, kernel_size=kernel_size, stride=stride, dilation=dilation_rate))
        self.blks.append(DilatedConvBLock1D(in_channels // 16, 1, input_size=input_size, kernel_size=kernel_size, stride=stride, dilation=1))
        
    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blks:
            x = blk(x)
        return x

class DilatedConv_Out_128(Module):
    def __init__(self, in_channels, dilation_rate, input_size, kernel_size, stride):
        super().__init__()
        self.blks = torch.nn.ModuleList()
        rate = 1.0
        self.blks.append(DilatedConvBLock1D(in_channels, int(in_channels * rate), input_size=input_size, kernel_size=kernel_size, stride=stride, dilation=1))
        self.blks.append(DilatedConvBLock1D(int(in_channels * rate), int(in_channels * rate), input_size=input_size, kernel_size=kernel_size, stride=stride, dilation=dilation_rate))
        self.blks.append(DilatedConvBLock1D(int(in_channels * rate), in_channels, input_size=input_size, kernel_size=kernel_size, stride=stride, dilation=dilation_rate))

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blks:
            x = blk(x)
        return x