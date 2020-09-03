__all__ = ['is_match_states_batch_size', 'verify_nmt_model', 'verify_nmt_inference']

import numpy.testing as npt
import mxnet as mx
from mxnet.util import use_np


def is_match_states_batch_size(states, states_batch_axis, batch_size) -> bool:
    """Test whether the generated states have the specified batch size

    Parameters
    ----------
    states
        The states structure
    states_batch_axis
        The states batch axis structure
    batch_size
        The batch size

    Returns
    -------
    ret
    """
    if states_batch_axis is None:
        return True
    if isinstance(states_batch_axis, int):
        if states.shape[states_batch_axis] == batch_size:
            return True
    for ele_states_batch_axis, ele_states in zip(states_batch_axis, states):
        ret = is_match_states_batch_size(ele_states, ele_states_batch_axis, batch_size)
        if ret is False:
            return False
    return True


@use_np
def verify_nmt_model(model, batch_size: int = 4,
                     src_seq_length: int = 5,
                     tgt_seq_length: int = 10,
                     atol: float = 1E-4,
                     rtol: float = 1E-4):
    """Verify the correctness of an NMT model. Raise error message if it detects problems.

    Parameters
    ----------
    model
        The machine translation model
    batch_size
        The batch size to test the nmt model
    src_seq_length
        Length of the source sequence
    tgt_seq_length
        Length of the target sequence
    atol
        Absolute tolerance.
    rtol
        Relative tolerance.

    """
    src_word_sequence = mx.np.random.randint(0, model.src_vocab_size, (batch_size, src_seq_length))
    tgt_word_sequence = mx.np.random.randint(0, model.tgt_vocab_size, (batch_size, tgt_seq_length))
    src_valid_length = mx.np.random.randint(1, src_seq_length, (batch_size,))
    min_tgt_seq_length = max(1, tgt_seq_length - 5)
    tgt_valid_length = mx.np.random.randint(min_tgt_seq_length, tgt_seq_length, (batch_size,))

    if model.layout == 'NT':
        full_out = model(src_word_sequence, src_valid_length, tgt_word_sequence, tgt_valid_length)
    else:
        full_out = model(src_word_sequence.T, src_valid_length,
                         tgt_word_sequence.T, tgt_valid_length)
        full_out = mx.np.swapaxes(full_out, 0, 1)
    if full_out.shape != (batch_size, tgt_seq_length, model.tgt_vocab_size):
        raise AssertionError('The output of NMT model does not match the expected output.'
                             ' Model output shape = {}, Expected (B, T, V) = {}'
                             .format(full_out.shape,
                                     (batch_size, tgt_seq_length, model.tgt_vocab_size)))
    for partial_batch_size in range(1, batch_size + 1):
        for i in range(1, min_tgt_seq_length):
            if model.layout == 'NT':
                partial_out = model(src_word_sequence[:partial_batch_size, :],
                                    src_valid_length[:partial_batch_size],
                                    tgt_word_sequence[:partial_batch_size, :(-i)],
                                    tgt_valid_length[:partial_batch_size]
                                    - mx.np.array(i, dtype=tgt_valid_length.dtype))
            else:
                partial_out = model(src_word_sequence[:partial_batch_size, :].T,
                                    src_valid_length[:partial_batch_size],
                                    tgt_word_sequence[:partial_batch_size, :(-i)].T,
                                    tgt_valid_length[:partial_batch_size]
                                    - mx.np.array(i, dtype=tgt_valid_length.dtype))
                partial_out = mx.np.swapaxes(partial_out, 0, 1)
            # Verify that the partial output matches the full output
            for b in range(partial_batch_size):
                partial_vl = tgt_valid_length.asnumpy()[b] - i
                npt.assert_allclose(full_out[b, :partial_vl].asnumpy(),
                                    partial_out[b, :partial_vl].asnumpy(), atol, rtol)


@use_np
def verify_nmt_inference(train_model, inference_model,
                         batch_size=4, src_seq_length=5,
                         tgt_seq_length=10, atol=1E-4, rtol=1E-4):
    """Verify the correctness of an NMT inference model. Raise error message if it detects
    any problems.

    Parameters
    ----------
    train_model
    inference_model
    batch_size
    src_seq_length
    tgt_seq_length
    atol
        Absolute tolerance
    rtol
        Relative tolerance

    """
    if train_model.layout == 'NT':
        src_word_sequences = mx.np.random.randint(0, train_model.src_vocab_size,
                                                  (batch_size, src_seq_length))
        tgt_word_sequences = mx.np.random.randint(0, train_model.tgt_vocab_size,
                                                  (batch_size, tgt_seq_length))
    else:
        src_word_sequences = mx.np.random.randint(0, train_model.src_vocab_size,
                                                  (src_seq_length, batch_size))
        tgt_word_sequences = mx.np.random.randint(0, train_model.tgt_vocab_size,
                                                  (tgt_seq_length, batch_size))
    src_valid_length = mx.np.random.randint(1, src_seq_length, (batch_size,))
    min_tgt_seq_length = max(1, tgt_seq_length - 5)
    tgt_valid_length = mx.np.random.randint(min_tgt_seq_length, tgt_seq_length, (batch_size,))
    full_out = train_model(src_word_sequences, src_valid_length,
                           tgt_word_sequences, tgt_valid_length)
    if train_model.layout == 'NT':
        for partial_batch_size in range(1, batch_size + 1):
            step_out_l = []
            states = inference_model.init_states(src_word_sequences[:partial_batch_size, :],
                                                 src_valid_length[:partial_batch_size])
            assert is_match_states_batch_size(states, inference_model.state_batch_axis,
                                              partial_batch_size)
            for i in range(min_tgt_seq_length):
                step_out, states = inference_model(tgt_word_sequences[:partial_batch_size, i],
                                                   states)
                step_out_l.append(step_out)
            partial_out = mx.np.stack(step_out_l, axis=1)
            npt.assert_allclose(full_out[:partial_batch_size, :min_tgt_seq_length].asnumpy(),
                                partial_out[:partial_batch_size, :].asnumpy(), atol, rtol)
    elif train_model.layout == 'TN':
        for partial_batch_size in range(1, batch_size + 1):
            step_out_l = []
            states = inference_model.init_states(src_word_sequences[:, :partial_batch_size],
                                                 src_valid_length[:partial_batch_size])
            assert is_match_states_batch_size(states, inference_model.state_batch_axis,
                                              partial_batch_size)
            for i in range(min_tgt_seq_length):
                step_out, states = inference_model(tgt_word_sequences[i, :partial_batch_size],
                                                   states)
                step_out_l.append(step_out)
            partial_out = mx.np.stack(step_out_l, axis=0)
            npt.assert_allclose(full_out[:min_tgt_seq_length, :partial_batch_size].asnumpy(),
                                partial_out[:, :partial_batch_size].asnumpy(), atol, rtol)
    else:
        raise NotImplementedError
