from transformer import model
import mxnet as mx
import io
import gluonnlp as nlp
import data


from XLNet_classifier import XLNetClassifier
modelname = 'xlnet_cased_l12_h768_a12'
ctx = mx.cpu(0)
use_decoder = False
xlnet_base, vocab, tokenizer = model.get_model(modelname, dataset_name='126gb', use_decoder=use_decoder)
#print(type(xlnet_base))
xlnet_classifier = XLNetClassifier(xlnet_base,ctx, num_classes=2, dropout=0.1)
xlnet_classifier.classifier.initialize(init = mx.init.Normal(0.02), ctx = ctx)
xlnet_classifier.pooler.initialize(init = mx.init.Normal(0.02), ctx = ctx)
#xlnet_classifier.hybridize(static_alloc = True)

#loss
loss = mx.gluon.loss.SoftmaxCELoss()
loss.hybridize(static_alloc = True)

metric = mx.metric.Accuracy()

tsv_file = io.open('dev.tsv', encoding='utf-8')

# Skip the first line, which is the schema
num_discard_samples = 1
# Split fields by tabs
field_separator = nlp.data.Splitter('\t')
# Fields to select from the file
field_indices = [3, 4, 0]
data_train_raw = nlp.data.TSVDataset(filename='dev.tsv',
                                 field_separator=field_separator,
                                 num_discard_samples=num_discard_samples,
                                 field_indices=field_indices)

# Use the vocabulary from pre-trained model for tokenization

xlnet_tokenizer = tokenizer

# The maximum length of an input sequence
max_len = 128

# The labels for the two classes [(0 = not similar) or  (1 = similar)]
all_labels = ["0", "1"]

# whether to transform the data as sentence pairs.
# for single sentence classification, set pair=False
# for regression task, set class_labels=None
# for inference without label available, set has_label=False
pair = True
transform = data.transform.XLNetDatasetTransform(xlnet_tokenizer, max_len,vocab= vocab,
                                                class_labels=all_labels,
                                                has_label=True,
                                                pad=True,
                                                pair=pair)

data_train = data_train_raw.transform(transform)



print('vocabulary used for tokenization = \n%s'%vocab)
print('%s token id = %s'%(vocab.padding_token, vocab[vocab.padding_token]))
print('%s token id = %s'%(vocab.cls_token, vocab[vocab.cls_token]))
print('%s token id = %s'%(vocab.sep_token, vocab[vocab.sep_token]))

# The hyperparameters
batch_size = 32
lr = 5e-6
bptt = 128


# The FixedBucketSampler and the DataLoader for making the mini-batches
train_sampler = nlp.data.FixedBucketSampler(lengths=[int(item[1]) for item in data_train],
                                            batch_size=batch_size,
                                            shuffle=True)

# eval_batchify = nlp.data.batchify.CorpusBPTTBatchify(vocab, bptt, batch_size,
#                                                          last_batch='discard')

bert_dataloader = mx.gluon.data.DataLoader(data_train, batch_sampler=train_sampler)

trainer = mx.gluon.Trainer(xlnet_classifier.collect_params(), 'adam',
                           {'learning_rate': lr, 'epsilon': 1e-9})

# Collect all differentiable parameters
# `grad_req == 'null'` indicates no gradients are calculated (e.g. constant parameters)
# The gradients for these params are clipped later
params = [p for p in xlnet_classifier.collect_params().values() if p.grad_req != 'null']
grad_clip = 1

# Training the model with only three epochs
log_interval = 4
num_epochs = 3
print("start_training....")
#eval_batchify = nlp.data.batchify.CorpusBPTTBatchify()
for epoch_id in range(num_epochs):
    metric.reset()
    step_loss = 0
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(bert_dataloader):
        print("batch_id: ", batch_id)
        print("toekn_id 0:", token_ids[0])
        print("segment_id: ", segment_ids[0])
        with mx.autograd.record():
            # Load the data to the GPU

            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = segment_ids.as_in_context(ctx)
            label = label.as_in_context(ctx)

            # Forward computation
            out = xlnet_classifier(token_ids, segment_ids, valid_length=valid_length.astype('float32'))
            ls = loss(out, label).mean()

        # And backwards computation
        ls.backward()

        # Gradient clipping
        trainer.allreduce_grads()
        nlp.utils.clip_grad_global_norm(params, 1)
        trainer.update(1)

        step_loss += ls.asscalar()
        metric.update([label], [out])

        # Printing vital information
        if (batch_id + 1) % (log_interval) == 0:
            print('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f}'
                         .format(epoch_id, batch_id + 1, len(bert_dataloader),
                                 step_loss / log_interval,
                                 trainer.learning_rate, metric.get()[1]))
            step_loss = 0

