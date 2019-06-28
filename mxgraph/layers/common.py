from mxnet.gluon import nn, HybridBlock, Block


class IdentityActivation(HybridBlock):
    def hybrid_forward(self, F, x):
        return x


class ELU(HybridBlock):
    r"""
    Exponential Linear Unit (ELU)
        "Fast and Accurate Deep Network Learning by Exponential Linear Units", Clevert et al, 2016
        https://arxiv.org/abs/1511.07289
        Published as a conference paper at ICLR 2016
    Parameters
    ----------
    alpha : float
        The alpha parameter as described by Clevert et al, 2016
    Inputs:
        - **data**: input tensor with arbitrary shape.
    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, alpha=1.0, **kwargs):
        super(ELU, self).__init__(**kwargs)
        self._alpha = alpha

    def hybrid_forward(self, F, x):
        return - self._alpha * F.relu(1.0 - F.exp(x)) + F.relu(x)


def get_activation(act):
    """Get the activation based on the act string

    Parameters
    ----------
    act: str or HybridBlock

    Returns
    -------
    ret: HybridBlock
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            return nn.LeakyReLU(0.1)
        elif act == 'identity':
            return IdentityActivation()
        elif act == 'elu':
            return ELU()
        elif act in ['relu', 'sigmoid', 'tanh', 'softrelu', 'softsign']:
            return nn.Activation(act)
        else:
            raise NotImplementedError
    else:
        return act
