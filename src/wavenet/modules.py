import torch


class CausalConvolution1D(torch.nn.Module):
    """Causal convolution module of WaveNet. 
    Causal filter is implemented as shifting and padding signal and then undo the shifting.

    Attributes:
        conv1d (torch.nn.Conv1d): 1D convolution layer
    """
    def __init__(self, input_channels, output_channels):
        """Initializes causal filter

        Args:
            input_channels (int): number of channels in the input
            output_channels (int): number of channels in the output
        """
        super(CausalConvolution1D, self).__init__()

        self.conv1d = torch.nn.Conv1d(input_channels, output_channels,
                                      kernel_size=2, stride=1, padding=1,
                                      bias=False)
        
    def forward(self, x):
        """Forward pass of causal convolution module, containing undo shifting.

        Args:
            x (tensor): input data
        
        Returns:
            Output of causal filter.
        """
        output = self.conv1d(x)
        return output[:, :, :-1]


class DilatedConvolution1D(torch.nn.Module):
    """Dilated convolution module of WaveNet.
    Performs dilated convolution on the input.

    Attributes:
        dilated_conv1d (torch.nn.Conv1d): 1D convolution layer
    """
    def __init__(self, channels, dilation):
        """Initializes dilated convolution layer

        Args:
            channels (int): number of channels in both input and output
            dilation (int): value of the dilation param
        """
        super(DilatedConvolution1D, self).__init__()

        self.dilated_conv1d = torch.nn.Conv1d(channels, channels,
                                              kernel_size=2, stride=1, padding=0,
                                              dilation=dilation, bias=False)

    def forward(self, x):
        """Forward pass of dilated convolution module.

        Args:
            x (tensor): input data
        
        Returns:
            Output of dilated convolution.
        """
        output = self.dilated_conv1d(x)
        return output


if __name__ == '__main__':
    inp = torch.randn(1, 20, 155)
    causal_conv = CausalConvolution1D(20, 100)
    out = causal_conv(inp)
    print(out.size())

    dilated_conv = DilatedConvolution1D(100, 2)
    out2 = dilated_conv(out)
    print(out.size())