from typing import Union, Tuple
import torch
from torch import Tensor, channel_shuffle
from torch.nn import Module, Conv1d, BatchNorm1d, ReLU, LeakyReLU

class DilatedConvBLock1D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_size: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        groups=1,
    ) -> None:
        """Dilated convolution block.

        :param in_channels: Amount of input channels.
        :type in_channels: int
        :param out_channels: Amount of output channels.
        :type out_channels: int
        :param kernel_size: Kernel shape.
        :type kernel_size: int|(int, int)
        :param stride: Stride shape.
        :type stride: int|(int, int)
        :param padding: Padding shape.
        :type padding: int|(int, int)
        :param dilation: Dilation shape.
        :type dilation: int|(int, int)
        """
        super().__init__()
        assert groups == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.input_size = input_size
        padding = self.get_padding_bins(input_size, self.dilation)
        # print(padding)
        self.cnn1 = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
            groups=groups,
        )

        self.batch_norm1 = BatchNorm1d(num_features=out_channels)

        # self.non_linearity = LeakyReLU(inplace=True)
        self.non_linearity = ReLU(inplace=True)

        self.cnn2 = Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
            groups=groups,
        )

        self.batch_norm2 = BatchNorm1d(num_features=out_channels)

    def get_padding_bins(self, input_length, dilations):
        return int(
            (
                input_length * (self.stride - 1)
                - self.stride
                + dilations * (self.kernel_size - 1)
                + 1
            )
            / 2
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the dilated\
        convolution block.

        :param x: Input.
        :type x: torch.Tensor
        :return: Output.
        :rtype: torch.Tensor
        """
        x = self.batch_norm1(self.non_linearity(self.cnn1(x)))
        x = self.batch_norm2(self.non_linearity(self.cnn2(x))) + x
        return x

class DilatedConv(Module):
    def __init__(
        self,
        in_channels: int,
        dilation_rate: int,
        input_size: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        """Dilated convolution block.

        :param in_channels: Amount of input channels.
        :type in_channels: int
        :param out_channels: Amount of output channels.
        :type out_channels: int
        :param kernel_size: Kernel shape.
        :type kernel_size: int|(int, int)
        :param stride: Stride shape.
        :type stride: int|(int, int)
        :param padding: Padding shape.
        :type padding: int|(int, int)
        :param dilation: Dilation shape.
        :type dilation: int|(int, int)
        """
        super().__init__()

        self.blks = torch.nn.ModuleList()

        if in_channels == 128:
            self.blks.append(
                DilatedConvBLock1D(
                    in_channels,
                    in_channels // 2,
                    input_size=input_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=1,
                )
            )
            self.blks.append(
                DilatedConvBLock1D(
                    in_channels // 2,
                    in_channels // 4,
                    input_size=input_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation_rate,
                )
            )
            self.blks.append(
                DilatedConvBLock1D(
                    in_channels // 4,
                    in_channels // 8,
                    input_size=input_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation_rate,
                )
            )
            self.blks.append(
                DilatedConvBLock1D(
                    in_channels // 8,
                    in_channels // 16,
                    input_size=input_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation_rate,
                )
            )
            self.blks.append(
                DilatedConvBLock1D(
                    in_channels // 16,
                    1,
                    input_size=input_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=1,
                )
            )
        elif in_channels == 40 or in_channels == 64:
            self.blks.append(
                DilatedConvBLock1D(
                    in_channels,
                    in_channels,
                    input_size=input_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=1,
                )
            )
            self.blks.append(
                DilatedConvBLock1D(
                    in_channels,
                    in_channels // 2,
                    input_size=input_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation_rate,
                )
            )
            self.blks.append(
                DilatedConvBLock1D(
                    in_channels // 2,
                    in_channels // 4,
                    input_size=input_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation_rate,
                )
            )
            self.blks.append(
                DilatedConvBLock1D(
                    in_channels // 4,
                    in_channels // 4,
                    input_size=input_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation_rate,
                )
            )
            self.blks.append(
                DilatedConvBLock1D(
                    in_channels // 4,
                    1,
                    input_size=input_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=1,
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the dilated\
        convolution block.

        :param x: Input.
        :type x: torch.Tensor
        :return: Output.
        :rtype: torch.Tensor
        """
        for blk in self.blks:
            x = blk(x)
        return x

class DilatedConv_Out_128(Module):
    def __init__(
        self,
        in_channels: int,
        dilation_rate: int,
        input_size: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        """Dilated convolution block.

        :param in_channels: Amount of input channels.
        :type in_channels: int
        :param out_channels: Amount of output channels.
        :type out_channels: int
        :param kernel_size: Kernel shape.
        :type kernel_size: int|(int, int)
        :param stride: Stride shape.
        :type stride: int|(int, int)
        :param padding: Padding shape.
        :type padding: int|(int, int)
        :param dilation: Dilation shape.
        :type dilation: int|(int, int)
        """
        super().__init__()

        self.blks = torch.nn.ModuleList()

        rate = 1.0
        if in_channels == 128:
            self.blks.append(
                DilatedConvBLock1D(
                    in_channels,
                    int(in_channels * rate),
                    input_size=input_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=1,
                )
            )
            self.blks.append(
                DilatedConvBLock1D(
                    int(in_channels * rate),
                    int(in_channels * rate),
                    input_size=input_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation_rate,
                )
            )
            # self.blks.append(DilatedConvBLock1D(int(in_channels * rate ** 2), int(in_channels * rate ** 2), input_size=input_size, kernel_size=kernel_size, stride=stride, dilation=dilation_rate))
            # self.blks.append(DilatedConvBLock1D(int(in_channels * rate ** 2), int(in_channels * rate), input_size=input_size, kernel_size=kernel_size, stride=stride, dilation=dilation_rate))
            self.blks.append(
                DilatedConvBLock1D(
                    int(in_channels * rate),
                    in_channels,
                    input_size=input_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation_rate,
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the dilated\
        convolution block.

        :param x: Input.
        :type x: torch.Tensor
        :return: Output.
        :rtype: torch.Tensor
        """
        for blk in self.blks:
            x = blk(x)
        return x