# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import time
import argparse

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_nli, get_batch, build_vocab
from mutils import get_optimizer
from models import NLINet
#from IPython import embed
import loadData as lD
from IPython import embed

GLOVE_PATH = "dataset/GloVe/glove.840B.300d.txt"


parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--nlipath", type=str, default='dataset/SNLI/true/', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--outputdir", type=str, default='newdir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model-finetune.pickle')

# training
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model
parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

# gpu
parser.add_argument("--gpu_id", type=int, default=3, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")

parser.add_argument("--toy", type=bool, default=False, help="toy")


params, _ = parser.parse_known_args()

# set gpu device
torch.cuda.set_device(params.gpu_id)

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)


"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
DATA
"""

toy = params.toy

"""
# toy SNLI provided, download the rest
GLOVE_PATH = './Downloads/glove.840B.300d.txt'
MODEL_PATH = './Downloads/infersent.allnli.pickle'
REGR_MODEL_PATH = './models/'
# If not None, where you want SNLI embeddings to be stored (WARNING: high memory)
EMBED_STORE = None
TEST_OUT_PATH = './regout/'

if toy:
    DATA_PATH = './Downloads/SNLI/toy/'
    REGR_MODEL_PATH = REGR_MODEL_PATH + 'TOY'
    EMBED_STORE = None
    TEST_OUT_PATH = TEST_OUT_PATH + 'TOY'
    TEST_DATA_PATH = './testData/toy/'
else:
    DATA_PATH = './Downloads/SNLI/true/'
    TEST_DATA_PATH = './testData/true/'

outpaths = {'REGR_MODEL_PATH': REGR_MODEL_PATH, 'TEST_OUT_PATH': TEST_OUT_PATH, 'TEST_DATA_PATH' : TEST_DATA_PATH}


"""
TEST_DATA_PATH = './testData/true/'
TEST_OUT_PATH = './regout/'
REGR_MODEL_PATH = './models/'
outpaths = {'REGR_MODEL_PATH': REGR_MODEL_PATH, 'TEST_OUT_PATH': TEST_OUT_PATH, 'TEST_DATA_PATH' : TEST_DATA_PATH}

id2label = {0:'CONTRADICTION', 1:'NEUTRAL', 2:'ENTAILMENT'}
label2id = {'CONTRADICTION': 0, 'NEUTRAL':1, 'ENTAILMENT':2}

#tasks = ['adjr', 'comp', 'ncon', 'subjv', 'temp', 'verb']
#tasks = [f[7:] for f in os.listdir(outpaths['TEST_DATA_PATH']) if 'label' in f]
tasks = ["comp_ml_long", "comp_ml_short"]

# Load scramble data in same format as snli (suitable for training)
scramble_data_path = './testData/toy/' if toy else './testData/true/'
scramble_train = lD.load_scramble_all("./testData/new/train/", label2id, tasks, one_group=True)
scramble_valid = lD.load_scramble_all("./testData/new/valid/", label2id, tasks, one_group=True)
scramble_test = lD.load_scramble_all("./testData/new/test/", label2id, tasks, one_group=True)

""" Convert Compositional Dataset format to SNLI dataset """
def convert_format(data, limit=5000):
    new_data = {}
    new_data["s1"] = data["X_A"][:limit]
    new_data["s2"] = data["X_B"][:limit]
    new_data["label"] = np.array(data["y"][:limit])

    return new_data

def convert_format_limit(old_data, limit=5000):
    new_data = old_data
    new_data["s1"] = new_data["s1"][:limit]
    new_data["s2"] = new_data["s2"][:limit]
    new_data["label"] = new_data["label"][:limit]
    return new_data

train_snli, valid_snli, test_snli = get_nli(params.nlipath)
train_comp = convert_format(scramble_train)
valid_comp = convert_format(scramble_valid)
test_comp = convert_format(scramble_test)


word_vec = build_vocab(train_snli['s1'] + train_snli['s2'] +
                       valid_snli['s1'] + valid_snli['s2'] +
                       test_snli['s1'] + test_snli['s2'] +
                       train_comp['s1'] + train_comp['s2'] +
                       valid_comp['s1'] + valid_comp['s2'] +
                       test_comp['s1'] + test_comp['s2'], GLOVE_PATH)

for split in ['s1', 's2']:
    for data_type in ['train_snli', 'valid_snli', 'test_snli', 'train_comp', 'valid_comp', 'test_comp']:
        eval(data_type)[split] = np.array([['<s>'] +
            [word for word in sent.split() if word in word_vec] +
            ['</s>'] for sent in eval(data_type)[split]])

params.word_emb_dim = 300


"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  len(word_vec)          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,

}

# model
encoder_types = ['BLSTMEncoder', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)
nli_net = NLINet(config_nli_model)
MODEL_PATH = './savedir/model.pickle'
nli_net = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
# nli_net = torch.load(MODEL_PATH)

# loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn.size_average = False

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(nli_net.parameters(), **optim_params)

# cuda by default
nli_net.cuda()
loss_fn.cuda()


"""
TRAIN
"""
val_acc_best = -1e10
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None


def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    # shuffle the data
    permutation = np.random.permutation(len(train_comp['s1']))

    s1 = train_comp['s1'][permutation]
    s2 = train_comp['s2'][permutation]
    target = train_comp['label'][permutation]


    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    for stidx in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],
                                     word_vec)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size],
                                     word_vec)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()
        k = s1_batch.size(1)  # actual batch size

        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params.batch_size])

        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.data[0])
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in nli_net.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm)

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if len(all_costs) == 100:
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                            stidx, round(np.mean(all_costs), 2),
                            int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            round(100.*correct/(stidx+k), 2)))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = round(100 * correct/len(s1), 2)
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))
    return train_acc


def evaluate(epoch, eval_type='valid', final_eval=False):
    nli_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop


    s1 = eval(eval_type)['s1']
    s2 = eval(eval_type)['s2']
    target = eval(eval_type)['label']

    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()

        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    # save model
    eval_acc = round(100 * correct / len(s1), 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))

    if eval_type == 'valid_comp' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(nli_net, os.path.join(params.outputdir,
                       params.outputmodelname))
            #embed()
            val_acc_best = eval_acc
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True

    print("Evaluate %s %.3lf" % (eval_type, eval_acc))
    return eval_acc


"""
Train model on Natural Language Inference task
"""
epoch = 1

print("Initial Evaluation (pre-training)")
train_acc = evaluate(epoch, "train_comp")
test_snli_acc = evaluate(epoch, "test_snli")
test_comp_acc = evaluate(epoch, "test_comp")

while not stop_training and epoch <= params.n_epochs:
    train_acc = trainepoch(epoch)

    train_acc = evaluate(epoch, "train_comp")
    est_snli_acc = evaluate(epoch, "test_snli")
    test_comp_acc = evaluate(epoch, "test_comp")
    #print "train_acc=", train_acc, " valid_acc(old)=", eval_acc, " valid_acc(new)=", test_acc
    epoch += 1

# Run best model on test set.
del nli_net
nli_net = torch.load(os.path.join(params.outputdir, params.outputmodelname))

print('\nTEST : Epoch {0}'.format(epoch))
train_acc = evaluate(1e6, "train_comp")
test_snli_acc = evaluate(1e6, "test_snli")
test_comp_acc = evaluate(1e6, "test_comp")

# Save encoder instead of full model
torch.save(nli_net.encoder,
           os.path.join(params.outputdir, params.outputmodelname + '.encoder'))
