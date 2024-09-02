import numpy as np
import torch
from collections import OrderedDict

from epde.evaluators import CustomEvaluator, EvaluatorTemplate, sign_evaluator, \
     trigonometric_evaluator
from epde.interface.prepared_tokens import PreparedTokens
from epde.interface.token_family import TokenFamily
import epde.globals as global_var

from epde.interface.prepared_tokens import GridTokens, ControlVarTokens, VarTrigTokens, DerivSignFunction

class DerivSignFunction(PreparedTokens):
    def __init__(self, var_name: str, token_labels: list):
        """
        Class for tokens, representing arbitrary functions of the modelled variable passed in `var_name` or its derivatives.  
        """
        token_type = f'signum of {var_name}`s derivatives'
        max_order = len(token_labels)

        deriv_solver_orders: list = [[0,]*(order+1) for order in range(max_order)]
        params_ranges = OrderedDict([('power', (1, 1))])
        params_equality_ranges = {'power': 0}

        self._token_family = TokenFamily(token_type = token_type, variable = var_name,
                                         family_of_derivs=True)

        self._token_family.set_status(demands_equation=False, meaningful=True,
                                      unique_specific_token=True, unique_token_type=True,
                                      s_and_d_merged=False, non_default_power = False)

        self._token_family.set_params(token_labels, params_ranges, params_equality_ranges,
                                      derivs_solver_orders=deriv_solver_orders)
        self._token_family.set_evaluator(sign_evaluator)


class VarTrigTokens(PreparedTokens):
    """
    Class for prepared tokens, that belongs to the trigonometric family
    """
    def __init__(self, var_name: str, freq_center: float = 1., 
                 max_power: int = 2, freq_eps = 1e-8):
        """
        Initialization of class

        Args:

        """
        freq = (freq_center - freq_eps, freq_center + freq_eps)

        self._token_family = TokenFamily(token_type=f'trigonometric of {var_name}', variable=var_name,
                                         family_of_derivs=True)
        self._token_family.set_status(demands_equation=False, unique_specific_token=True, unique_token_type=True,
                                      meaningful=True, non_default_power = True)
            
        def latex_form(label, **params):
            '''
            Parameters
            ----------
            label : str
                label of the token, for which we construct the latex form.
            **params : dict
                dictionary with parameter labels as keys and tuple of parameter values 
                and their output text forms as values.

            Returns
            -------
            form : str
                LaTeX-styled text form of token.
            '''
            form = label + r'^{{{0}}}'.format(params["power"][1]) + \
                    r'(' + params["freq"][1] + r' x_{' + params["dim"][1] + r'})'
            return form
        
        self._token_family.set_latex_form_constructor(latex_form)
        trig_token_params = OrderedDict([('power', (1, max_power)),
                                         ('freq', freq)])
        
        trig_equal_params = {'power': 0, 'freq': 2*freq_eps}

        adapted_labels = [f'sin({var_name})', f'cos({var_name})']
        deriv_solver_orders = [[None,] for label in adapted_labels]

        def trig_sine(*args, **kwargs):
            return np.sin(kwargs['freq'] * args[0]) ** kwargs['power']
        
        def trig_cosine(*args, **kwargs):
            return np.cos(kwargs['freq'] * args[0]) ** kwargs['power']

        def torch_trig_sine(*args, **kwargs):
            return torch.sin(kwargs['freq'] * args[0]) ** kwargs['power']
        
        def torch_trig_cosine(*args, **kwargs):
            return torch.cos(kwargs['freq'] * args[0]) ** kwargs['power']

        trig_eval_fun_np = {adapted_labels[0]: trig_sine,
                            adapted_labels[1]: trig_cosine}

        trig_eval_fun_torch = {adapted_labels[0]: torch_trig_sine,
                            adapted_labels[1]: torch_trig_cosine}        

        eval = CustomEvaluator(evaluation_functions_np = trig_eval_fun_np,
                               evaluation_functions_torch = trig_eval_fun_torch,
                               eval_fun_params_labels = ['power', 'freq'])

        self._token_family.set_params(adapted_labels, trig_token_params, trig_equal_params, 
                                      derivs_solver_orders=deriv_solver_orders)
        self._token_family.set_evaluator(eval)