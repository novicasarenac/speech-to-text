import torch
from wavenet.modules import *


class Wavenet(torch.nn.Module):
    """Wavenet implementation.

    Attributes:
        causal_convolutions (CausalConvolution1D): Causal convolutions 1d module.
        residual_stack (ResidualStack): Residual stack module of Wavenet.
        softmax_layer (SoftmaxLayer): Last layer of Wavenet.
        receptive_fields_num (int): Total number of receptive fields in Wavenet.
    """
    def __init__(self, input_channels, stack_len, dilations_per_layer,
                 res_layers_num, softmax_output):
        """Initializes Wavenet:

        Args:
            input_channels (int): Number of channels in the input.
            stack_len (int): Number of layers in residual stack.
            dilations_per_layer (int): Number of dilations (residual blocks) per layer in residual stack.
            res_layers_num (int): number of both input and output channels in single residual block
            softmax_output (int): size of softmax output (vocabulary length)
        """
        super(Wavenet, self).__init__()
        
        self.causal_convolutions = CausalConvolution1D(input_channels,
                                                       res_layers_num)
        self.residual_stack = ResidualStack(stack_len, dilations_per_layer,
                                            res_layers_num, input_channels)
        self.softmax_layer = SoftmaxLayer(input_channels, softmax_output)
        self.receptive_fields_num = self._count_receptive_fields(stack_len,
                                                            dilations_per_layer)

    def _count_receptive_fields(self, stack_len, dilations_per_layer):
        """Counts total number of receptive fields in the network

        Args:
            stack_len (int): Number of layers in residual stack.
            dilations_per_layer (int): Number of dilations (residual blocks) per layer in residual stack.

        Returns:
            Total number of receptive fields.
        """
        return int(sum([2 ** i for i in range(0, dilations_per_layer)]) * stack_len)

    def _calc_skip_size(self, x):
        """Calculates skip size for residual stack.

        Args:
            x (tensor): Input data.

        Returns:
            Skip size.
        """
        if int(x.size(2)) < self.receptive_fields_num:
            return None
        else:
            return int(x.size(2)) - self.receptive_fields_num

    def forward(self, x):
        """Forward pass of Wavenet.

        Args:
            x (tensor): Input data.
        
        Returns:
            Output of Wavenet.
        """
        skip_size = self._calc_skip_size(x)
        if skip_size:
            causal_conv_output = self.causal_convolutions(x)
            res_stack_output = self.residual_stack(causal_conv_output, skip_size)
            output = self.softmax_layer(res_stack_output)
            
            return output.transpose(1, 2).contiguous()
        else:
            return torch.IntTensor()