r"""
This file contains some useful function and class.
"""

import json
import math

try:
    from config import DATA_PATH, INIT_LEARNING_RATE, WARM_UP_STEPS
except ImportError:
    from .config import DATA_PATH, INIT_LEARNING_RATE, WARM_UP_STEPS


def mask_logits(x, mask):
    r"""Implement mask logits computation.

        Parameters
        -----------
        x : NDArray
            input tensor with shape `(batch_size, sequence_length)`
        mask : NDArray
            input tensor with shape `(batch_size, sequence_length)`

        Returns
        --------
        return : NDArray
            output tensor with shape `(batch_size, sequence_length)`
        """
    return x + -1e30 * (1 - mask)


def load_emb_mat(file_name):
    r"""Implement load embedding matrix.

        Parameters
        -----------
        file_name : string
            the embedding matrix file name.

        Returns
        --------
        mat : List[List]
            output 2-D list.
        """
    with open(DATA_PATH + file_name) as f:
        mat = json.loads(f.readline())
    return mat


def warm_up_lr(step):
    r"""Implement learning rate warm up.

        Parameters
        -----------
        step : int
            control the learning rate linear increase.

        Returns
        --------
        return : int
            the learning rate for next weight update.
        """
    return min(INIT_LEARNING_RATE, INIT_LEARNING_RATE * (math.log(step) / math.log(WARM_UP_STEPS)))


def zero_grad(params):
    r"""
    Set the grad to zero.
    """
    for _, paramter in params.items():
        paramter.zero_grad()


class ExponentialMovingAverage():
    r"""An implement of Exponential Moving Average.

        shadow variable = decay * shadow variable + (1 - decay) * variable

    Parameters
    ----------
    decay : float, default 0.9999
        The axis to sum over when computing softmax and entropy.
    """

    def __init__(self, decay=0.9999, **kwargs):
        super(ExponentialMovingAverage, self).__init__(**kwargs)
        self.decay = decay
        self.shadow = {}

    def add(self, name, parameters):
        r"""Update the shadow variable.

        Parameters
        -----------
        name : string
            the name of shadow variable.
        parameters : NDArray
            the init value of shadow variable.
        Returns
        --------
        return : None
        """
        self.shadow[name] = parameters.copy()

    def __call__(self, name, x):
        r"""Update the shadow variable.

        Parameters
        -----------
        name : string
            the name of shadow variable.
        x : NDArray
            the value of shadow variable.
        Returns
        --------
        return : None
        """
        assert name in self.shadow
        self.shadow[name] = self.decay * \
            self.shadow[name] + (1.0 - self.decay) * x

    def get(self, name):
        r"""Return the shadow variable.

        Parameters
        -----------
        name : string
            the name of shadow variable.

        Returns
        --------
        return : NDArray
            the value of shadow variable.
        """
        return self.shadow[name]
