import ast
import os
import inspect
import logging
import re
import mxnet.ndarray as nd
from mxnet import gluon
import mxnet as mx
import numpy as np


def safe_eval(expr):
    if type(expr) is str:
        return ast.literal_eval(expr)
    else:
        return expr


def logging_config(folder=None, name=None,
                   level=logging.DEBUG,
                   console_level=logging.DEBUG,
                   no_console=True):
    """

    Parameters
    ----------
    folder : str or None
    name : str or None
    level : int
    console_level
    no_console: bool
        Whether to disable the console log

    Returns
    -------

    """
    if name is None:
        name = inspect.stack()[1][1].split('.')[0]
    if folder is None:
        folder = os.path.join(os.getcwd(), name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Remove all the current handlers
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".log")
    print("All Logs will be saved to %s" %logpath)
    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)
    if not no_console:
        # Initialze the console logging
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder


def parse_ctx(ctx_args):
    ctx = re.findall('([a-z]+)(\d*)', ctx_args)
    ctx = [(device, int(num)) if len(num) > 0 else (device, 0) for device, num in ctx]
    ctx = [mx.Context(*ele) for ele in ctx]
    return ctx


def gluon_total_param_num(net):
    return sum([np.prod(v.shape) for v in net.collect_params().values()])


def gluon_net_info(net, save_path=None):
    info_str = 'Total Param Number: {}\n'.format(gluon_total_param_num(net)) +\
               'Params:\n'
    for k, v in net.collect_params().items():
        info_str += '\t{}: {}, {}\n'.format(k, v.shape, np.prod(v.shape))
    info_str += str(net)
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(info_str)
    return info_str


def params_clip_global_norm(param_dict, clip, ctx):
    grads = [p.grad(ctx) for p in param_dict.values()]
    gnorm = gluon.utils.clip_global_norm(grads, clip)
    return gnorm


def get_global_norm(arrays):
    ctx = arrays[0].context
    total_norm = nd.add_n(*[nd.dot(x, x).as_in_context(ctx)
                            for x in (arr.reshape((-1,)) for arr in arrays)])
    total_norm = nd.sqrt(total_norm).asscalar()
    return total_norm

