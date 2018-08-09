"""Extensions to NLP Toolkit Data Stream API."""

import mxnet as mx
import numpy as np

from gluonnlp.data import DataStream


class BucketingStream(DataStream):
    """Transform a stream of batches into bucketed batches.

    Parameters
    ----------
    stream : DataStream
        Stream of list of list/tuple of integers (a stream over shards of
        the dataset).
    split : int
        Number of batches to return for each incoming batch.
    length_fn : callable
        Callable to determine the length of each batch dimension in the
        input batch. The resulting array of lengths is used as sort key for
        the buckets.
    batchify_fn : callable
        Extract a bucket batch given selected indices and the input batch.

    """

    def __init__(self, stream, split, length_fn, batchify_fn):
        self._stream = stream
        self._split = split
        self._length_fn = length_fn
        self._batchify_fn = batchify_fn

    def __iter__(self):
        for input_batch in self._stream:
            lengths = self._length_fn(input_batch)
            if isinstance(lengths, mx.nd.NDArray):
                lengths = lengths.asnumpy()
            sorted_lengths = np.argsort(lengths)
            splits = np.array_split(sorted_lengths, self._split)
            for split in splits:
                if len(split):
                    yield self._batchify_fn(split, input_batch)
