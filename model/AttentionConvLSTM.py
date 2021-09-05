#=====================================================================================
#  Author: Aobo Li
#  Contact: liaobo77@gmail.com
#  
#  Last Modified: Aug. 29, 2021
#  
#  * This file contains Convolutional LSTM module enhanced with attention mechanism. 
#  * Prototype code from https://github.com/ndrplz/ConvLSTM_pytorch
#  * Attention is added into the original code
#  * returns the context image instead of standard LSTM output (output,(hidden,cell))
#=====================================================================================
import torch.nn as nn
import torch
import torchsnooper
from torch.nn.parameter import Parameter

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        print(self.input_dim + self.hidden_dim, 4 * self.hidden_dim)

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B(batch size), T(time channel), C(hidden state channel), H(height), W(width) or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, time_channel,
                 batch_first=False, bias=True, return_all_layers=False, return_hidden_and_context = False, fill_value=0.1):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.return_hidden_and_context = return_hidden_and_context

        # Initialize the attention weight of attention mechanism, and fill it with random values between (-fill_value, fill_value)
        # The dimmension of attention weight is (T, C, H, W) (remember that H=W in our case)
        self.attention_weight = Parameter(torch.empty(time_channel[0], hidden_dim[-1], time_channel[1],time_channel[1]).uniform_(-fill_value, fill_value))

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        self.input_array = 0
        self.output_score = 0

    #@torchsnooper.snoop()
    def forward(self, input_tensor, hidden_state=None, att=False):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        # Using the attention mechanism to produce context images
        hs = layer_output_list[-1]                                                  # (B, T, C, H, W)
        ht = last_state_list[-1][0]                                                 # (B, C, H, W)
        w_attention = self.attention_weight.unsqueeze(0).expand(*hs.size())         # (T, C, H, W) -> (B, T, C, H, W)
        ht_input = ht.unsqueeze(1).expand(*hs.size())                               # (B, C, H, W) -> (B, T, C, H, W)
        score_input = hs*w_attention*ht_input                                       # (B, T, C, H, W)
        score_input = score_input.permute(0,2,1,3,4)                                # (B, T, C, H, W) -> (B, C, T, H, W)
        hs = hs.permute(0,2,1,3,4)
        score = torch.softmax(score_input,dim=2)                                    # (B, C, T, H, W) -> (B, C, H, W)
        context = torch.sum(score * hs,dim=2)                                       # (B, C, T, H, W) -> (B, C, H, W)

        output = context
        if self.return_hidden_and_context:
            '''
            Return context vector for training/validation
            '''
            output = torch.cat([context,ht],dim=1)
        if att:
            '''
            Return the attention score for network interpretability study
            '''
            self.input_array, self.output_score = self.return_attention_score(input_tensor, score)
            self.input_array = self.input_array.detach()
            self.output_score = self.output_score.detach()
            return self.input_array, self.output_score
        return output

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states
    # @torchsnooper.snoop()
    def return_attention_score(self,input_event,score):
        '''
        Read out the total attention score for given input events
        '''
        b, t, _, _, _ = input_event.size()
        input_array = input_event.view(b,t,-1)
        input_array = torch.sum(input_array,dim=-1)
        

        dim0, dim1, dim2 = (score.size(0),score.size(1),score.size(2))
        output_score = score.view(dim0,dim1, dim2,-1)
        output_score = torch.sum(output_score,dim=-1)
        return input_array, output_score


    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param