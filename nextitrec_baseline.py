import torch
import torch.nn as nn
from generator_recsys import NextItNet_Decoder
import utils
import shutil
import time
import math
import numpy as np
import argparse
import Data_loader
import os
import random

# You can run it directly, first training and then evaluating
# nextitrec_generate.py can only be run when the model parameters are saved, i.e.,
#  save_path = saver.save(sess,
#                       "Data/Models/generation_model/model_nextitnet.ckpt".format(iter, numIters))
# if you are dealing very huge industry dataset, e.g.,several hundred million items, you may have memory problem during training, but it 
# be easily solved by simply changing the last layer, you do not need to calculate the cross entropy loss
# based on the whole item vector. Similarly, you can also change the last layer (use tf.nn.embedding_lookup or gather) in the prediction phrase 
# if you want to just rank the recalled items instead of all items. The current code should be okay if the item size < 5 million.



#Strongly suggest running codes on GPU with more than 10G memory!!!
#if your session data is very long e.g, >50, and you find it may not have very strong internal sequence properties, you can consider generate subsequences
def generatesubsequence(train_set):
    # create subsession only for training
    subseqtrain = []
    for i in range(len(train_set)):
        # print x_train[i]
        seq = train_set[i]
        lenseq = len(seq)
        # session lens=100 shortest subsession=5 realvalue+95 0
        for j in range(lenseq - 2):
            subseqend = seq[:len(seq) - j]
            subseqbeg = [0] * j
            subseq = np.append(subseqbeg, subseqend)
            # beginseq=padzero+subseq
            # newsubseq=pad+subseq
            subseqtrain.append(subseq)
    x_train = np.array(subseqtrain)  # list to ndarray
    del subseqtrain
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_train = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffle_train]
    print("generating subsessions is done!")
    return x_train


def INFO_LOG(info):
    print("[%s]%s"%(time.strftime("%Y-%m-%d %X", time.localtime()), info))

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def getBatch(data, batch_size):
	start_inx = 0
	end_inx = batch_size

	while end_inx < len(data):
		batch = data[start_inx:end_inx]
		start_inx = end_inx
		end_inx += batch_size
		yield batch

	# if end_inx >= len(data):
	# 	batch = data[start_inx:]
	# 	yield batch


parser = argparse.ArgumentParser()
parser.add_argument('--top_k', type=int, default=5,
                    help='Sample from top k predictions')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='hyperpara-Adam')
parser.add_argument('--batch_size', default=128, type=int)
# history_sequences_20181014_fajie
# ml20m_update_ls30gr5
# mllatest_update_ls100gr3.csv
parser.add_argument('--datapath', type=str, default='Data/Session/ml20m_update_ls30gr5.csv',
                    help='data path')
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--savedir', default='Data/checkpoint', type=str)
parser.add_argument('--tt_percentage', type=float, default=0.2,
                    help='0.2 means 80% training 20% testing')
parser.add_argument('--is_generatesubsession', type=bool, default=False,
                    help='whether generating a subsessions, e.g., 12345-->01234,00123,00012  It may be useful for very some very long sequences')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--shrink_lr', action="store_true", default=False)
parser.add_argument('--L2', default=0, type=float)
args = parser.parse_args()
print(args)
dl = Data_loader.Data_Loader({'model_type': 'generator', 'dir_name': args.datapath})
all_samples = dl.items
items_voc = dl.item2id

print("shape: ", np.shape(all_samples))

# Split train/test set
dev_sample_index = -1 * int(args.tt_percentage * float(len(all_samples)))
train_set, valid_set = all_samples[:dev_sample_index], all_samples[dev_sample_index:]

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(train_set)))
train_set = train_set[shuffle_indices]


if args.is_generatesubsession:
    x_train = generatesubsequence(train_set)

model_para = {
    #if you changed the parameters here, also do not forget to change paramters in nextitrec_generate.py
    'item_size': len(items_voc),
    'dilated_channels': 256,
    # if you use nextitnet_residual_block, you can use [1, 4, ],
    # if you use nextitnet_residual_block_one, you can tune and i suggest [1, 2, 4, ], for a trial
    # when you change it do not forget to change it in nextitrec_generate.py
    'dilations': [1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4],
    'kernel_size': 3,
    'batch_size':args.batch_size,
    'iterations':200,
    'is_negsample':False, #False denotes no negative sampling
    'seq_len': len(all_samples[0]),
    'pad': dl.padid,
}
print("dilations", model_para["dilations"])
print("dilated_channels", model_para["dilated_channels"])
print("batch_size", model_para["batch_size"])

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = NextItNet_Decoder(model_para).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

if args.shrink_lr == True:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.02)

criterion = nn.CrossEntropyLoss()
curr_preds_5 = []
rec_preds_5 = []
ndcg_preds_5 = []
curr_preds_20 = []
rec_preds_20 = []
ndcg_preds_20 = []
best_acc = 0


def test(epoch):
    global best_acc
    model.eval()
    # test_loss = 0
    correct = 0
    total = 0
    batch_size = model_para['batch_size']
    batch_num = valid_set.shape[0] / batch_size
    INFO_LOG("-------------------------------------------------------test")
    with torch.no_grad():
        start = time.time()
        for batch_idx, batch_sam in enumerate(getBatch(valid_set, batch_size)):
            inputs, targets = torch.LongTensor(batch_sam[:,:-1]).to(args.device), torch.LongTensor(batch_sam[:,-1]).to(args.device).view([-1])
            outputs = model(inputs, onecall=True) # [batch_size, item_size] only predict the last position

            accuracy(outputs.data.cpu().numpy(), targets.data.cpu().numpy(), 0, batch_idx, batch_num, epoch)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        end = time.time()
        print('Acc(hit@1): %.3f%% (%d/%d)' % (100. * correct / total, correct, total))
        INFO_LOG("TIME FOR EPOCH During Testing: {}".format(end - start))
        INFO_LOG("TIME FOR BATCH (mins): {}".format((end - start) / batch_num))
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        state = {
            'net': model.state_dict(),
            'acc(hit@1)': acc
        }
        torch.save(state, '%s/best_weishi_%s.t7' % (args.savedir, model_para['dilations']))
    print('epoch:%d    accuracy(hit@1):%.3f    best:%.3f' % (epoch, acc, best_acc))

    INFO_LOG("epoch: {}\t total_epoch:{}\t total_batches:{}".format(
        epoch, args.epochs, batch_num))
    INFO_LOG("Accuracy mrr_5: {}".format(sum(curr_preds_5) / float(len(curr_preds_5))))
    INFO_LOG("Accuracy mrr_20: {}".format(sum(curr_preds_20) / float(len(curr_preds_20))))
    INFO_LOG("Accuracy hit_5: {}".format(sum(rec_preds_5) / float(len(rec_preds_5))))
    INFO_LOG("Accuracy hit_20: {}".format(sum(rec_preds_20) / float(len(rec_preds_20))))
    INFO_LOG("Accuracy ndcg_5: {}".format(sum(ndcg_preds_5) / float(len(ndcg_preds_5))))
    INFO_LOG("Accuracy ndcg_20: {}".format(sum(ndcg_preds_20) / float(len(ndcg_preds_20))))


def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_size = model_para['batch_size']
    batch_num = train_set.shape[0] / batch_size
    start = time.time()
    INFO_LOG("-------------------------------------------------------train")
    for batch_idx, batch_sam in enumerate(getBatch(train_set, batch_size)):
        inputs, targets = torch.LongTensor(batch_sam[:, :-1]).to(args.device), torch.LongTensor(batch_sam[:, 1:]).to(
            args.device).view([-1])
        optimizer.zero_grad()
        outputs = model(inputs) # [batch_size*seq_len, item_size]
        loss = criterion(outputs, targets)

        L2_loss = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                L2_loss += torch.norm(param, 2)
        loss += args.L2 * L2_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % max(10, batch_num//10) == 0:
            INFO_LOG("epoch: {}\t {}/{}".format(epoch, batch_idx, batch_num))
            print('Loss: %.3f | Acc(hit@1): %.3f%% (%d/%d)' % (
                train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    end = time.time()
    INFO_LOG("TIME FOR EPOCH During Training: {}".format(end - start))
    INFO_LOG("TIME FOR BATCH (mins): {}".format((end - start) / batch_num))
    if args.shrink_lr:
        lr_scheduler.step()


def accuracy(output, target, loss, batch_idx, batch_num, epoch, topk=(args.top_k, args.top_k+15)): # output: [batch_size, item_size] target: [batch_size]
    """Computes the accuracy over the k top predictions for the specified values of k"""
    global curr_preds_5
    global rec_preds_5
    global ndcg_preds_5
    global curr_preds_20
    global rec_preds_20
    global ndcg_preds_20

    for bi in range(output.shape[0]):
        pred_items_5 = utils.sample_top_k(output[bi], top_k=topk[0])  # top_k=5
        pred_items_20 = utils.sample_top_k(output[bi], top_k=topk[1])

        true_item=target[bi]
        predictmap_5={ch : i for i, ch in enumerate(pred_items_5)}
        pred_items_20 = {ch: i for i, ch in enumerate(pred_items_20)}

        rank_5 = predictmap_5.get(true_item)
        rank_20 = pred_items_20.get(true_item)
        if rank_5 == None:
            curr_preds_5.append(0.0)
            rec_preds_5.append(0.0)
            ndcg_preds_5.append(0.0)
        else:
            MRR_5 = 1.0/(rank_5+1)
            Rec_5 = 1.0#3
            ndcg_5 = 1.0 / math.log(rank_5 + 2, 2)  # 3
            curr_preds_5.append(MRR_5)
            rec_preds_5.append(Rec_5)#4
            ndcg_preds_5.append(ndcg_5)  # 4
        if rank_20 == None:
            curr_preds_20.append(0.0)
            rec_preds_20.append(0.0)#2
            ndcg_preds_20.append(0.0)#2
        else:
            MRR_20 = 1.0/(rank_20+1)
            Rec_20 = 1.0#3
            ndcg_20 = 1.0 / math.log(rank_20 + 2, 2)  # 3
            curr_preds_20.append(MRR_20)
            rec_preds_20.append(Rec_20)#4
            ndcg_preds_20.append(ndcg_20)  # 4

    if batch_idx % max(10, batch_num//10) == 0:
        # INFO_LOG("epoch/total_epoch: {}/{}\t batch/total_batches: {}/{} \t loss: {:.3f}".format(
        #             epoch, args.epochs, batch_idx,  batch_num, loss/(batch_idx+1)))
        INFO_LOG("epoch/total_epoch: {}/{}\t batch/total_batches: {}/{}".format(
            epoch, args.epochs, batch_idx, batch_num))

        INFO_LOG("Accuracy hit_5: {}".format(sum(rec_preds_5) / float(len(rec_preds_5))))  # 5
        INFO_LOG("Accuracy hit_20: {}".format(sum(rec_preds_20) / float(len(rec_preds_20))))  # 5



if __name__ == '__main__':
    for i, (key, u) in enumerate(model.state_dict().items()):
        print(key, u.size())
    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)
        state = {
            'net': model.state_dict(),
        }
        torch.save(state, '%s/ckpt_%d.t7' % (args.savedir, epoch))

