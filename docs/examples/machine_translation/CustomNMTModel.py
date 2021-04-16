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
"""class ONNXNMTModel, used in transformer_onnx_based.md"""

import numpy as np
import mxnet as mx
import onnxruntime

class ONNXNMTModel:
    """This class mimics the actual NMTModel class defined here:
    https://github.com/dmlc/gluon-nlp/blob/v0.10.0/src/gluonnlp/model/translation.py#L28
    """
    class ONNXRuntimeSession:
        """This class is used to wrap the onnxruntime sessions of the components in the
        transforman model, namely: src_embed, encoder, tgt_embed, one_step_ahead_decoder,
        and tgt_proj.
        """
        def __init__(self, onnx_file):
            """Init the onnxruntime session. Performace tuning code can be added here.
            Parameters
            ----------
            onnx_file : str
            """
            ses_opt = onnxruntime.SessionOptions()
            ses_opt.log_severity_level = 3
            self.session = onnxruntime.InferenceSession(onnx_file, ses_opt)
            
        def __call__(self, *onnx_inputs):
            """Notice that the inputs here are MXNet NDArrays. We first convert them to numpy
            ndarrays, run inference, and then convert the outputs back to MXNet NDArrays.
            Parameters
            ----------
            onnx_inputs: list of NDArrays
            Returns
            -------
            list of NDArrays
            """
            input_dict = dict((self.session.get_inputs()[i].name, onnx_inputs[i].asnumpy())
                            for i in range(len(onnx_inputs)))
            outputs = self.session.run(None, input_dict)
            if len(outputs) == 1:
                return mx.nd.array(outputs[0])
            return [mx.nd.array(output) for output in outputs]
    
    class DummyDecoder:
        """This Dummy Decoder mimics the actualy decoder defined here:
        https://github.com/dmlc/gluon-nlp/blob/v0.10.0/src/gluonnlp/model/transformer.py#L724
        For inference we only need to define init_state_from_encoder()
        """
        def init_state_from_encoder(self, encoder_outputs, encoder_valid_length=None):
            """Initialize the state from the encoder outputs. Refer to the original function here:
            https://github.com/dmlc/gluon-nlp/blob/v0.10.0/src/gluonnlp/model/transformer.py#L621
            Parameters
            ----------
            encoder_outputs : list
            encoder_valid_length : NDArray or None
            Returns
            -------
            decoder_states : list
                The decoder states, includes:
                - mem_value : NDArray
                - mem_masks : NDArray or None
            """
            mem_value = encoder_outputs
            decoder_states = [mem_value]
            mem_length = mem_value.shape[1]
            if encoder_valid_length is not None:
                dtype = encoder_valid_length.dtype
                ctx = encoder_valid_length.context
                mem_masks = mx.nd.broadcast_lesser(
                    mx.nd.arange(mem_length, ctx=ctx, dtype=dtype).reshape((1, -1)),
                    encoder_valid_length.reshape((-1, 1)))
                decoder_states.append(mem_masks)
            else:
                decoder_states.append(None)
            return decoder_states

    def __init__(self, tgt_vocab, src_embed_onnx_file, encoder_onnx_file, tgt_embed_onnx_file,
                 one_step_ahead_decoder_onnx_file, tgt_proj_onnx_file):
        """Init the ONNXNMTModel. For inference we need the following components of the original
        transformer model: src_embed, encoder, tgt_embed, one_step_ahead_decoder, and tgt_proj.
        Parameters
        ----------
        tgt_vocab : Vocab
            Target vocabulary.
        src_embed_onnx_file: str
        encoder_onnx_file: str
        tgt_embed_onnx_file: str
        one_step_ahead_decoder_onnx_file: str
        tgt_proj_onnx_file: str
        """
        self.tgt_vocab = tgt_vocab
        self.src_embed = self.ONNXRuntimeSession(src_embed_onnx_file)
        self.encoder = self.ONNXRuntimeSession(encoder_onnx_file)
        self.tgt_embed = self.ONNXRuntimeSession(tgt_embed_onnx_file)
        self.one_step_ahead_decoder = self.ONNXRuntimeSession(one_step_ahead_decoder_onnx_file)
        self.tgt_proj = self.ONNXRuntimeSession(tgt_proj_onnx_file)
        self.decoder = self.DummyDecoder()
    
    def encode(self, inputs, states=None, valid_length=None):
        """Encode the input sequence. Refer to the original function here:
        https://github.com/dmlc/gluon-nlp/blob/v0.10.0/src/gluonnlp/model/translation.py#L132
        Parameters
        ----------
        inputs : NDArray
        states : list of NDArrays or None, default None
        valid_length : NDArray or None, default None
        Returns
        -------
        outputs : list
            Outputs of the encoder.
        """
        return self.encoder(self.src_embed(inputs), valid_length), None
        
    def decode_step(self, step_input, decoder_states):
        """One step decoding of the translation model. Refer to the original function here:
        https://github.com/dmlc/gluon-nlp/blob/v0.10.0/src/gluonnlp/model/translation.py#L171
        Parameters
        ----------
        step_input : NDArray
            Shape (batch_size,)
        states : list of NDArrays
        Returns
        -------
        step_output : NDArray
            Shape (batch_size, C_out)
        states : list
        step_additional_outputs : list
            Additional outputs of the step, e.g, the attention weights
        """
        step_input = self.tgt_embed(step_input)

        # Refer to https://github.com/dmlc/gluon-nlp/blob/v0.10.0/src/gluonnlp/model/transformer.py#L819
        if len(decoder_states) == 3:  # step_input from prior call is included
            last_embeds, _, _ = decoder_states
            inputs = mx.nd.concat(last_embeds, mx.nd.expand_dims(step_input, axis=1), dim=1)
            decoder_states = decoder_states[1:]
        else:
            inputs = mx.nd.expand_dims(step_input, axis=1)

        # Refer to https://github.com/dmlc/gluon-nlp/blob/v0.10.0/src/gluonnlp/model/transformer.py#L834
        step_output = self.one_step_ahead_decoder(decoder_states[1], inputs, decoder_states[0])
        decoder_states = [inputs] + decoder_states
        step_additional_outputs = None

        step_output = self.tgt_proj(step_output)

        return step_output, decoder_states, step_additional_outputs