from abc import ABC
from typing import List, Tuple, Union
from collections import OrderedDict

import torch

class BasicDeriv(ABC):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Trying to create abstract differentiation method')
    
    def take_derivative(self, u: torch.Tensor, args: torch.Tensor, axes: list):
        raise NotImplementedError('Trying to differentiate with abstract differentiation method')


class AutogradDeriv(BasicDeriv):
    def __init__(self):
        pass

    def take_derivative(self, u: Union[torch.nn.Sequential, torch.Tensor], args: torch.Tensor, 
                        axes: list = [], component: int = 0):
        if not args.requires_grad:
            args.requires_grad = True
        if axes == [None,]:
            return u(args)[..., component].reshape(-1, 1)
        if isinstance(u, torch.nn.Sequential):
            comp_sum = u(args)[..., component].sum(dim = 0)
        elif isinstance(u, torch.Tensor):
            raise TypeError('Autograd shall have torch.nn.Sequential as its inputs.')
        else:
            print(f'u.shape, {u.shape}')
            comp_sum = u.sum(dim = 0)
        for axis in axes:
            output_vals = torch.autograd.grad(outputs = comp_sum, inputs = args, create_graph=True)[0]
            comp_sum = output_vals[:, axis].sum()
        output_vals = output_vals[:, axes[-1]].reshape(-1, 1)
        return output_vals

def prepare_control_inputs(model: torch.nn.Sequential, grid: torch.Tensor, 
                           args: List[Tuple[Union[int, List]]]) -> torch.Tensor:
    '''
    Recompute the control ANN arguments tensor from the solutions of 
    controlled equations $L \mathbf{u}(t, \mathbf{x}, \mathbf{c}) = 0$, 
    calculating necessary derivatives, as `args` 

    Args:
        model (`torch.nn.Sequential`): solution of the controlled equation $\mathbf{u}(\mathbf{u})$.
        
        grid (`torch.Tensor`): tensor of the grids m x n, where m - number of points in the domain, n - number of NN inputs.

        args (`List[Tuple[Union[int, List]]]`) - list of arguments of derivative operators.

    Returns:
        `torch.Tensor`: tensor of arguments for the control ANN.
    '''
    differntiatior = AutogradDeriv()
    ctrl_inputs = torch.cat([differntiatior.take_derivative(u = model, args = grid, axes = arg[1],
                                                            component = arg[0]).reshape(-1, 1) for arg in args], dim = 1)
    return ctrl_inputs

@torch.no_grad()
def eps_increment_diff(input_params: OrderedDict, loc: List[Union[str, Tuple[int]]], 
                       forward: bool = True, eps = 1e-4): # input_keys: list,  prev_loc: List = None, 
    if forward:
        input_params[loc[0]][tuple(loc[1:])] += eps  
    else:     
        input_params[loc[0]][tuple(loc[1:])] -= 2*eps
    return input_params