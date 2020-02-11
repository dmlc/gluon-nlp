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
import json
import logging

import mxnet as mx
import gluonnlp as nlp


class BertHandler:
    """GluonNLP based Bert Handler"""

    def __init__(self):
        self.error = None
        self._context = None
        self.initialized = False

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        gpu_id = context.system_properties["gpu_id"]
        self._mx_ctx = mx.cpu() if gpu_id is None else mx.gpu(gpu_id)
        bert, vocab = nlp.model.get_model('bert_12_768_12',
                                          dataset_name='book_corpus_wiki_en_uncased',
                                          pretrained=False, ctx=self._mx_ctx, use_pooler=True,
                                          use_decoder=False, use_classifier=False)
        tokenizer = nlp.data.BERTTokenizer(vocab, lower=True)
        self.sentence_transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=128,
                                                                 vocab=vocab, pad=True, pair=False)
        self.batchify = nlp.data.batchify.Tuple(
            nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token]),  # input
            nlp.data.batchify.Stack(),  # length
            nlp.data.batchify.Pad(axis=0, pad_val=0))  # segment
        # Set dropout to non-zero, to match pretrained model parameter names
        self.net = nlp.model.BERTClassifier(bert, dropout=0.1)
        self.net.load_parameters('sst.params', self._mx_ctx)
        self.net.hybridize()

        self.initialized = True

    def handle(self, batch, context):
        # we're just faking batch_size==1 but allow dynamic batch size. Ie the
        # actual batch size is the len of the first element.
        try:
            assert len(batch) == 1
            batch = json.loads(batch[0]["data"].decode('utf-8'))
        except (json.JSONDecodeError, KeyError, AssertionError) as e:
            print('call like: curl -X POST http://127.0.0.1:8080/bert_sst/predict '
                  '-F \'data=["sentence 1", "sentence 2"]\'')
            raise e
        model_input = self.batchify([self.sentence_transform(sentence) for sentence in batch])

        inputs, valid_length, token_types = [arr.as_in_context(self._mx_ctx) for arr in model_input]
        inference_output = self.net(inputs, token_types, valid_length.astype('float32'))
        inference_output = inference_output.as_in_context(mx.cpu())

        return [mx.nd.softmax(inference_output).argmax(axis=1).astype('int').asnumpy().tolist()]


_service = BertHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
