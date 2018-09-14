import json
import time

import mxnet as mx
from mxnet import autograd, gluon, nd
from tqdm import tqdm

from data_loader import DataLoader
from evaluate import evaluate

try:
    from config import *
except ImportError:
    from .config import *
try:
    from model import MySoftmaxCrossEntropy, QANet
except ImportError:
    from .model import MySoftmaxCrossEntropy, QANet
try:
    from util import ExponentialMovingAverage, load_emb_mat, warm_up_lr, zero_grad
except ImportError:
    from .util import ExponentialMovingAverage, load_emb_mat, warm_up_lr, zero_grad


mx.random.seed(37)
accum_avg_train_ce = []
batch_train_ce = []
dev_f1 = []
dev_em = []
global_step = LAST_GLOBAL_STEP


def train(model, data_loader, trainer, loss_function, ema=None):
    for e in tqdm(range(EPOCHS)):
        print('Begin %d/%d epoch...' % (e + 1, EPOCHS))
        train_one_epoch(model, data_loader, trainer, loss_function, ema)

        # save model after train one epoch
        model.save_parameters(SAVE_MODEL_PREFIX_NAME +
                              time.asctime(time.localtime(time.time())))
        trainer.save_states(SAVE_TRAINER_PREFIX_NAME +
                            time.asctime(time.localtime(time.time())))
        with open(ACCUM_AVG_TRAIN_CROSS_ENTROPY + time.asctime(time.localtime(time.time())), 'w') as f:
            f.write(json.dumps(accum_avg_train_ce))
        with open(BATCH_TRAIN_CROSS_ENTROPY + time.asctime(time.localtime(time.time())), 'w') as f:
            f.write(json.dumps(batch_train_ce))


def train_one_epoch(model, data_loader, trainer, loss_function, ema=None):
    total_batchs = data_loader.total_batchs
    total_loss = 0
    step = 0
    global global_step
    for batch_data in data_loader.next_batch():
        step += 1
        global_step += 1
        # add evaluate per EVALUATE_INTERVAL batchs
        if(global_step % EVALUATE_INTERVAL == 0):
            print('global_step == %d' % (global_step))
            print('evaluating dev dataset...')
            f1_score, em_score = evaluate(model, dataset_type='dev', ema=ema)
            print('dev f1:' + str(f1_score) + 'em:' + str(em_score))
            dev_f1.append([global_step, f1_score])
            dev_em.append([global_step, em_score])


        context = nd.array([x[0] for x in batch_data])
        query = nd.array([x[1] for x in batch_data])
        c_mask = context > 0
        q_mask = query > 0
        context_char = nd.array([x[2] for x in batch_data])
        query_char = nd.array([x[3] for x in batch_data])
        begin = nd.array([x[4] for x in batch_data])
        end = nd.array([x[5] for x in batch_data])
        batch_sizes = context.shape[0]
        context = gluon.utils.split_and_load(
            data=context,
            ctx_list=CTX,
            # even_split=False
        )
        c_mask = gluon.utils.split_and_load(
            data=c_mask,
            ctx_list=CTX,
            # even_split=False
        )

        query = gluon.utils.split_and_load(
            data=query,
            ctx_list=CTX,
            # even_split=False
        )
        q_mask = gluon.utils.split_and_load(
            data=q_mask,
            ctx_list=CTX,
            # even_split=False
        )
        context_char = gluon.utils.split_and_load(
            data=context_char,
            ctx_list=CTX,
            # even_split=False
        )
        query_char = gluon.utils.split_and_load(
            data=query_char,
            ctx_list=CTX,
            # even_split=False
        )
        begin = gluon.utils.split_and_load(
            data=begin,
            ctx_list=CTX,
            # even_split=False
        )
        end = gluon.utils.split_and_load(
            data=end,
            ctx_list=CTX,
            # even_split=False
        )

        with autograd.record():
            different_ctx_loss = [loss_function(*model(c, q, cc, qc, cm, qm, b, e)) for c, q, cc, qc, cm, qm, b, e in zip(
                context, query, context_char, query_char, c_mask, q_mask, begin, end)]

            for loss in different_ctx_loss:
                loss.backward()
        if global_step == 1:
            # TODO: multi-GPU
            for name, param in model.collect_params().items():
                ema.register(name, param.data(CTX[0]))
        trainer.set_learning_rate(warm_up_lr(global_step))
        trainer.allreduce_grads()
        reset_embedding_grad(model)
        tmp = []
        for name, paramater in model.collect_params().items():
            grad = paramater.grad(context[0].context)
            if name == "qanet0_embedding0_weight":
                grad[0:2] += WEIGHT_DECAY * \
                    paramater.data(context[0].context)[0:2]
            else:
                grad += WEIGHT_DECAY * paramater.data(context[0].context)
            tmp.append(grad)
        gluon.utils.clip_global_norm(tmp, CLIP_GRADIENT)
        reset_embedding_grad(model)
        trainer.update(batch_sizes, ignore_stale_grad=True)
        # zero_grad(model.collect_params())
        # trainer.step(TRAIN_BATCH_SIZE, ignore_stale_grad=True)
        for name, param in model.collect_params().items():
            ema(name, param.data(CTX[0]))

        # nd.waitall()
        batch_loss = .0
        for loss in different_ctx_loss:
            batch_loss += loss.mean().asscalar()
        batch_loss /= len(different_ctx_loss)
        total_loss += batch_loss

        batch_train_ce.append([global_step, batch_loss])
        accum_avg_train_ce.append([global_step, total_loss / step])

        print("batch %d/%d, total_loss %.2f, batch_loss %.2f" %
              (step, total_batchs, total_loss / step, batch_loss), end='\r', flush=True)


def initial_model_parameters(model):
    model.collect_params().initialize(ctx=CTX)
    # initial embedding parameters with glove
    word_embedding = nd.array(load_emb_mat(WORD_EMB_FILE_NAME))

    model.word_emb[0].weight.set_data(word_embedding)


def reset_embedding_grad(model):
    for ctx in CTX:
        model.word_emb[0].weight.grad(ctx=ctx)[2:] = 0


def main():
    model = QANet()
    # initial parameters
    print('Initial paramters...')
    if NEED_LOAD_TRAINED_MODEL:
        model.load_parameters(TARGET_MODEL_FILE_NAME, ctx=CTX)
    else:
        print('Initial model parameters...')
        # TODO: confirm in multi-gpu
        initial_model_parameters(model)
    print(model)
    if TRAIN_FLAG == True:
        loss_function = MySoftmaxCrossEntropy()

        ema = ExponentialMovingAverage(decay=EXPONENTIAL_MOVING_AVERAGE_DECAY)
        # ema = None
        # initial trainer
        trainer = gluon.Trainer(
            model.collect_params(),
            'adam',
            {
                # 'wd': WEIGHT_DECAY,
                'learning_rate': INIT_LEARNING_RATE,
                'beta1': BETA1,
                'beta2': BETA2,
                'epsilon': EPSILON,
                # 'clip_gradient': CLIP_GRADIENT
            }
        )

        if NEED_LOAD_TRAINED_MODEL:
            trainer.load_states(TARGET_TRAINER_FILE_NAME)

        # initial dataloader
        train_data_loader = DataLoader(
            batch_size=TRAIN_BATCH_SIZE, dev_set=False)

        # train
        print('Train...')
        train(model, train_data_loader, trainer, loss_function, ema)
    else:
        print('Evaluating dev set...')
        f1_score, em_score = evaluate(model, dataset_type='dev', ema=None)
        print('The dev dataset F1 is:%s, and EM is: %s' % (f1_score, em_score))


if __name__ == "__main__":
    main()
