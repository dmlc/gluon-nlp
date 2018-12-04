# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""A registry for datasets

The registry makes it simple to construct a dataset given its name.

"""
__all__ = ['register', 'create', 'list_datasets']

import inspect

from mxnet import registry
from mxnet.gluon.data import Dataset

_REGSITRY_NAME_KWARGS = {}


def register(class_=None, **kwargs):
    """Registers a dataset with segment specific hyperparameters.

    When passing keyword arguments to `register`, they are checked to be valid
    keyword arguments for the registered Dataset class constructor and are
    saved in the registry. Registered keyword arguments can be retrieved with
    the `list_datasets` function.

    All arguments that result in creation of separate datasets should be
    registered. Examples are datasets divided in different segments or
    categories, or datasets containing multiple languages.

    Once registered, an instance can be created by calling
    :func:`~gluonnlp.data.create` with the class name.

    Parameters
    ----------
    **kwargs : list or tuple of allowed argument values
        For each keyword argument, it's value must be a list or tuple of the
        allowed argument values.

    Examples
    --------
    >>> @gluonnlp.data.register(segment=['train', 'test', 'dev'])
    ... class MyDataset(gluon.data.Dataset):
    ...     def __init__(self, segment='train'):
    ...         pass
    >>> my_dataset = gluonnlp.data.create('MyDataset')
    >>> print(type(my_dataset))
    <class 'MyDataset'>

    """

    def _real_register(class_):
        # Assert that the passed kwargs are meaningful
        for kwarg_name, values in kwargs.items():
            try:
                real_args = inspect.getfullargspec(class_).args
            except AttributeError:
                # pylint: disable=deprecated-method
                real_args = inspect.getargspec(class_.__init__).args

            if not kwarg_name in real_args:
                raise RuntimeError(
                    ('{} is not a valid argument for {}. '
                     'Only valid arguments can be registered.').format(
                         kwarg_name, class_.__name__))

            if not isinstance(values, (list, tuple)):
                raise RuntimeError(('{} should be a list of '
                                    'valid arguments for {}. ').format(
                                        values, kwarg_name))

        # Save the kwargs associated with this class_
        _REGSITRY_NAME_KWARGS[class_] = kwargs

        register_ = registry.get_register_func(Dataset, 'dataset')
        return register_(class_)

    if class_ is not None:
        # Decorator was called without arguments
        return _real_register(class_)

    return _real_register


def create(name, **kwargs):
    """Creates an instance of a registered dataset.

    Parameters
    ----------
    name : str
        The dataset name (case-insensitive).

    Returns
    -------
    An instance of :class:`mxnet.gluon.data.Dataset` constructed with the
    keyword arguments passed to the create function.

    """
    create_ = registry.get_create_func(Dataset, 'dataset')
    return create_(name, **kwargs)


def list_datasets(name=None):
    """Get valid datasets and registered parameters.

    Parameters
    ----------
    name : str or None, default None
        Return names and registered parameters of registered datasets. If name
        is specified, only registered parameters of the respective dataset are
        returned.

    Returns
    -------
    dict:
        A dict of all the valid keyword parameters names for the specified
        dataset. If name is set to None, returns a dict mapping each valid name
        to its respective keyword parameter dict. The valid names can be
        plugged in `gluonnlp.model.word_evaluation_model.create(name)`.

    """
    reg = registry.get_registry(Dataset)

    if name is not None:
        class_ = reg[name.lower()]
        return _REGSITRY_NAME_KWARGS[class_]
    else:
        return {
            dataset_name: _REGSITRY_NAME_KWARGS[class_]
            for dataset_name, class_ in registry.get_registry(Dataset).items()
        }
