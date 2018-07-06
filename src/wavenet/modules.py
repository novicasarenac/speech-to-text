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


class ResidualBlock(torch.nn.Module):
    """Single residual block implementation.

    Attributes:
        dilated_convolution (DilatedConvolution1D): dilated convolution layer as part of residual
        residual_output_convolution (torch.nn.Conv1d): convolutional layer at the output of residual
        skip_connection_convolution (torch.nn.Conv1d): convolutional layer at skip connection side of residual
        tanh (torch.nn.Tanh): tanh layer
        sigmoid (toch.nn.Sigmoid): sigmoid layer
    """
    def __init__(self, res_layers_num, skip_layers_num, dilation):
        """Initializes residual block

        Args:
            res_layers_num (int): number of both input and output channels in residual block
            skip_layers_num (int): number of output layers in skip connection
            dilation (int): value of the dilation param
        """
        super(ResidualBlock, self).__init__()

        self.dilated_convolution = DilatedConvolution1D(res_layers_num,
                                                        dilation)
        self.residual_output_convolution = torch.nn.Conv1d(res_layers_num,
                                                           res_layers_num, 1)
        self.skip_connection_convolution = torch.nn.Conv1d(res_layers_num,
                                                           skip_layers_num, 1)
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x, skip_size):
        """Forward pass of single residual block.

        Args:
            x (tensor): input data
            skip_size (int): size of output data at skip connection output
        
        Returns:
            Output of residual block and output at skip connection side of
            residual block.
        """
        dilated = self.dilated_convolution(x)
        tanh_step = self.tanh(dilated)
        sigmoid_step = self.sigmoid(dilated)
        dilated_transformed = tanh_step * sigmoid_step

        # Output of residual
        residual_conv = self.residual_output_convolution(dilated_transformed)
        input_matrix = x[:, :, -residual_conv.size(2):]
        output = residual_conv + input_matrix

        # Skip connections side output
        skip_connection = self.skip_connection_convolution(dilated_transformed)
        skip_connection_output = skip_connection[:, :, -skip_size:]

        return output, skip_connection_output


class ResidualStack(torch.nn.Module):
    def __init__(self, stack_len, dilations_per_layer, res_layers_num,
                 skip_layers_num):
        """Initialize residual stack.

        Args:
            stack_len (int): Number of layers in residual stack
            dilations_per_layer (int): Number of dilations (residual blocks) per layer in residual stack
            res_layers_num (int): number of both input and output channels in single residual block
            skip_layers_num (int): number of output layers in skip connection in single residual block
        """
        super(ResidualStack, self).__init__()

        self.stack_len = stack_len
        self.dilations_per_layer = dilations_per_layer
        self.res_layers_num = res_layers_num
        self.skip_layers_num = skip_layers_num
        
        self.res_stack = initialize_res_stack()

    def initialize_res_stack(self):
        residual_blocks = []
        dilations = []
        for layer_ind in range(0, self.stack_len):
            dilations.extend([2 ** dilation_ind for dilation_ind in range(0, self.dilations_per_layer)])
        
        for dilation in dilations:
            residual_blocks.append(self._make_residual_block(dilation))
        
        return residual_blocks
    
    def _make_residual_block(self, dilation):
        residual_block = ResidualBlock(self.res_layers_num, self.skip_layers_num, dilation)
        # TODO


if __name__ == '__main__':
    inp = torch.randn(1, 20, 155)
    causal_conv = CausalConvolution1D(20, 100)
    out = causal_conv(inp)
    # print(out.size())

    # dilated_conv = DilatedConvolution1D(100, 2)
    # out2 = dilated_conv(out)
    # print(out.size())

    residual_block = ResidualBlock(100, 50, 2)
    output, skip_out = residual_block(out, 20)
    print(out.size())
    print(output.size())
    print(skip_out.size())