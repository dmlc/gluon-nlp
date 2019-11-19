"""XLNet dataset transform."""
__all__ = ['XLNetSentenceTransform', 'XLNetDatasetTransform']
import numpy as np

class XLNetSentenceTransform:
    r"""XLNet style data transformation.

    The transformation is processed in the following steps:
    - tokenize the input sequences
    - insert [CLS], [SEP] as necessary. Note that the [CLS] token is inserted
    to the end in XLNet(which is to the beginning in Bert)
    - generate type ids to indicate whether a token belongs to the first
      sequence or the second sequence.
    - generate valid length
    - pad the sequence to max_length. Note that we use left pad in XLNet

    For sequence pairs, the input is a tuple of 3 strings:
    text_a, text_b and label.
    Inputs:
        text_a: 'is this jacksonville ?'
        text_b: 'no it is not'
        label: '0'
    Tokenization:
        text_a: 'is this jack ##son ##ville ?'
        text_b: 'no it is not .'
    Processed:
        tokens:  'is this jack ##son ##ville ? [SEP] no it is not . [SEP] [CLS]'
        type_ids: 0  0    0    0     0       0 0     1  1  1  1   1 1      1
        valid_length: 14
        label: 0
    For single sequences, the input is a tuple of 2 strings: text_a and label.
    Inputs:
        text_a: 'the dog is hairy .'
        label: '1'
    Tokenization:
        text_a: 'the dog is hairy .'
    Processed:
        text_a:  'the dog is hairy . [SEP] [CLS]'
        type_ids: 0   0   0  0     0 0      0
        valid_length: 7
        label: 1

    Parameters
    ----------
    line: tuple of str
        Input strings. For sequence pairs, the input is a tuple of 3 strings:
        (text_a, text_b, label). For single sequences, the input is a tuple
        of 2 strings: (text_a, label).

    Returns
    -------
    np.array: input token ids in 'int32', shape (batch_size, seq_length)
    np.array: valid length in 'int32', shape (batch_size,)
    np.array: input token type ids in 'int32', shape (batch_size, seq_length)
    """

    def __init__(self, tokenizer, max_seq_length=None, vocab=None, pad=True, pair=True):
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad = pad
        self._pair = pair
        self._vocab = self._tokenizer.vocab if vocab is None else vocab
        # RoBERTa does not register CLS token and SEP token
        if hasattr(self._vocab, 'cls_token'):
            self._cls_token = self._vocab.cls_token
        else:
            self._cls_token = self._vocab.bos_token
        if hasattr(self._vocab, 'sep_token'):
            self._sep_token = self._vocab.sep_token
        else:
            self._sep_token = self._vocab.eos_token
        self._padding_token = self._vocab.padding_token

    def __call__(self, line):

        text_a = line[0]
        if self._pair:
            assert len(line) == 2
            text_b = line[1]

        tokens_a = self._tokenizer(text_a)
        tokens_b = None

        if self._pair:
            tokens_b = self._tokenizer(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b,
                                    self._max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self._max_seq_length - 2:
                tokens_a = tokens_a[0:(self._max_seq_length - 2)]

        # The embedding vectors for `type=0` and `type=1` were learned during
        # pre-training and are added to the embedding vector
        # (and position vector). This is not *strictly* necessary since
        # the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.

        # For classification tasks, the last vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        tokens.extend(tokens_a)
        tokens.append(self._sep_token)
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens.extend(tokens_b)
            tokens.append(self._sep_token)
            tokens.append(self._cls_token)
            segment_ids.extend([1] * (len(tokens) - len(segment_ids)))
        else:
            tokens.append(self._cls_token)
            segment_ids.extend([0])

        input_ids = self._vocab[tokens]

        # The valid length of sentences. Only real  tokens are attended to.
        valid_length = len(input_ids)

        if self._pad:
            # Zero-pad up to the sequence length.
            padding_length = self._max_seq_length - valid_length
            # use padding tokens for the rest
            input_ids = [self._vocab[self._padding_token]] * padding_length + input_ids
            segment_ids = [1] * padding_length + segment_ids

        return np.array(input_ids, dtype='int32'), np.array(valid_length, dtype='int32'), \
               np.array(segment_ids, dtype='int32')

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()


class XLNetDatasetTransform:
    """Dataset transformation for XLNet-style sentence classification or regression.

    Parameters
    ----------
    tokenizer : BERTTokenizer.
        Tokenizer for the sentences.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    vocab : Vocab or BERTVocab
        The vocabulary.
    labels : list of int , float or None. defaults None
        List of all label ids for the classification task and regressing task.
        If labels is None, the default task is regression
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    pair : bool, default True
        Whether to transform sentences or sentence pairs.
    label_dtype: int32 or float32, default float32
        label_dtype = int32 for classification task
        label_dtype = float32 for regression task
    """
    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 vocab=None,
                 class_labels=None,
                 label_alias=None,
                 pad=True,
                 pair=True,
                 has_label=True):
        self.class_labels = class_labels
        self.has_label = has_label
        self._label_dtype = 'int32' if class_labels else 'float32'
        if has_label and class_labels:
            self._label_map = {}
            for (i, label) in enumerate(class_labels):
                self._label_map[label] = i
            if label_alias:
                for key in label_alias:
                    self._label_map[key] = self._label_map[label_alias[key]]
        self._xl_xform = XLNetSentenceTransform(
            tokenizer, max_seq_length=max_seq_length, vocab=vocab, pad=pad, pair=pair)

    def __call__(self, line):
        """Perform transformation for sequence pairs or single sequences.

        The transformation is processed in the following steps:
        - tokenize the input sequences
        - insert [CLS], [SEP] as necessary
        - generate type ids to indicate whether a token belongs to the first
          sequence or the second sequence.
        - generate valid length

        For sequence pairs, the input is a tuple of 3 strings:
        text_a, text_b and label.
        Inputs:
            text_a: 'is this jacksonville ?'
            text_b: 'no it is not'
            label: '0'
        Tokenization:
            text_a: 'is this jack ##son ##ville ?'
            text_b: 'no it is not .'
        Processed:
            tokens:  'is this jack ##son ##ville ? [SEP] no it is not . [SEP] [CLS]'
            type_ids: 0  0    0    0     0       0 0     1  1  1  1   1 1      1
            valid_length: 14
            label: 0
        For single sequences, the input is a tuple of 2 strings: text_a and label.
        Inputs:
            text_a: 'the dog is hairy .'
            label: '1'
        Tokenization:
            text_a: 'the dog is hairy .'
        Processed:
            text_a:  'the dog is hairy . [SEP] [CLS]'
            type_ids: 0   0   0  0     0 0      0
            valid_length: 7
            label: 1

        Parameters
        ----------
        line: tuple of str
            Input strings. For sequence pairs, the input is a tuple of 3 strings:
            (text_a, text_b, label). For single sequences, the input is a tuple
            of 2 strings: (text_a, label).

        Returns
        -------
        np.array: input token ids in 'int32', shape (batch_size, seq_length)
        np.array: valid length in 'int32', shape (batch_size,)
        np.array: input token type ids in 'int32', shape (batch_size, seq_length)
        np.array: classification task: label id in 'int32', shape (batch_size, 1),
            regression task: label in 'float32', shape (batch_size, 1)
        """
        if self.has_label:
            input_ids, valid_length, segment_ids = self._xl_xform(line[:-1])
            label = line[-1]
            # map to int if class labels are available
            if self.class_labels:
                label = self._label_map[label]
            label = np.array([label], dtype=self._label_dtype)
            return input_ids, valid_length, segment_ids, label
        else:
            return self._xl_xform(line)
            