import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils import get_data
from src.model import GCN_CAPS_Model
from src.L2Regularization import Regularization
from src.eval_metrics import *

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 设置固定的随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

# 记录输出
class Logger(object):
    def __init__(self, filename='default.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def train(train_loader, model, criterion, optimizer, epoch, weight_decay, reg_loss, args):
    results = []
    truths = []
    model.train()
    total_loss = 0.0
    total_batch_size = 0

    for ind, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
        # measure data loading time
        sample_ind, text, audio, video = batch_X
        text, audio, video = text.cuda(non_blocking=True), audio.cuda(non_blocking=True), video.cuda(non_blocking=True)
        batch_Y = batch_Y.cuda(non_blocking=True)
        eval_attr = batch_Y.squeeze(-1)
        batch_size = text.size(0)
        total_batch_size += batch_size

        preds = model(text, audio, video, batch_size)
        # print('preds:{}', preds)
        # print('eval_attr:{}', eval_attr)
        if args.dataset in ['mosi', 'mosei_senti']:
            preds = preds.reshape(-1)
            eval_attr = eval_attr.reshape(-1)
            raw_loss = criterion(preds, eval_attr)
            # 错误样本分析，保存错误样本的预测值和真实值
            # wrong_preds, true_eval = wrong_sample(preds, eval_attr)

            if weight_decay > 0:
                raw_loss = raw_loss + reg_loss(model)

            results.append(preds)
            truths.append(eval_attr)
        elif args.dataset == 'iemocap':
            preds = preds.view(-1, 2)
            eval_attr = eval_attr.view(-1)
            raw_loss = criterion(preds, eval_attr)
            if weight_decay > 0:
                raw_loss = raw_loss + reg_loss(model)
            results.append(preds)
            truths.append(eval_attr)

        total_loss += raw_loss.item() * batch_size
        combined_loss = raw_loss
        optimizer.zero_grad()
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) # 梯度裁剪，防止梯度爆炸
        optimizer.step()

    avg_loss = total_loss / total_batch_size
    results = torch.cat(results)
    truths = torch.cat(truths)
    return avg_loss, results, truths


def validate(loader, model, criterion, args):
    model.eval()
    results = []
    truths = []
    total_loss = 0.0
    total_batch_size = 0
    with torch.no_grad():
        for ind, (batch_X, batch_Y, batch_META) in enumerate(loader):
            sample_ind, text, audio, video = batch_X
            text, audio, video = text.cuda(non_blocking=True), audio.cuda(non_blocking=True), video.cuda(non_blocking=True)
            batch_Y = batch_Y.cuda(non_blocking=True)
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
            batch_size = text.size(0)
            total_batch_size += batch_size
            preds = model(text, audio, video, batch_size)
            if args.dataset in ['mosi', 'mosei_senti']:
                preds = preds.reshape(-1)
                eval_attr = eval_attr.reshape(-1)
                raw_loss = criterion(preds, eval_attr)
                results.append(preds)
                truths.append(eval_attr)
                total_loss += raw_loss.item() * batch_size
            elif args.dataset == 'iemocap':
                preds = preds.view(-1, 2)
                eval_attr = eval_attr.view(-1)
                raw_loss = criterion(preds, eval_attr)
                results.append(preds)
                truths.append(eval_attr)
                total_loss += raw_loss.item() * batch_size

    avg_loss = total_loss / total_batch_size
    results = torch.cat(results)
    truths = torch.cat(truths)
    return avg_loss, results, truths


if __name__ == "__main__":
    setup_seed(2021)
    sys.stdout = Logger('result.txt', sys.stdout)
    sys.stderr = Logger('error.txt', sys.stderr)
    parser = argparse.ArgumentParser(description='PyTorch GCN_CAPS Learner')
    parser.add_argument('--aligned', action='store_true', default=False, help='consider aligned experiment or not')
    parser.add_argument('--dataset', type=str, default='mosi', help='dataset to use')
    parser.add_argument('--data-path', type=str, default='MULT-dataset', help='path for storing the dataset')
    parser.add_argument('--epochs', default=1, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=32, type=int) #32
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                        help='initial learning rate', dest='lr')
    parser.add_argument('--MULT_d', default=30, type=int, help='the output dimensionality of MULT is 2*MULT_d')
    parser.add_argument('--vertex_num', default=20, type=int, help='number of vertexes') #20
    parser.add_argument('--dim_capsule', default=32, type=int, help='dimension of capsule') #32
    parser.add_argument('--routing', default=3, type=int, help='total routing rounds') #3
    parser.add_argument('--weight_decay', default=0.001, type=float, help='L2Regularization')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout in primary capsule in StoG')
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--clip', type=float, default=1, help='gradient clip value (default: 1)')
    parser.add_argument('--patience', default=10, type=int, help='patience for learning rate decay')
    parser.add_argument('--layers', default=3, type=int, help='the number of layer used')
    parser.add_argument('--k_1', default=8, type=int, help='the number of adjacent nodes when constructing the graph')
    parser.add_argument('--k_2', default=8, type=int, help='the number of adjacent nodes when constructing the hypergraph')
    args = parser.parse_args()

    assert args.dataset in ['mosi', 'mosei_senti', 'iemocap'], "supported datasets are mosei_senti, mosi and iemocap"

    hyp_params = args
    hyp_params.MULT_d = args.MULT_d
    hyp_params.vertex_num = args.vertex_num
    hyp_params.dim_capsule = args.dim_capsule
    hyp_params.routing = args.routing
    hyp_params.weight_decay = args.weight_decay
    hyp_params.dropout = hyp_params.dropout # 为什么不是=args.dropout
    hyp_params.layers = args.layers
    hyp_params.k_1 = args.k_1
    hyp_params.k_2 = args.k_2
    hyp_params.batch_size = args.batch_size
    current_setting = (hyp_params.MULT_d, hyp_params.vertex_num, hyp_params.dim_capsule, hyp_params.routing,
                       hyp_params.dropout, hyp_params.weight_decay, hyp_params.layers, hyp_params.k_1, hyp_params.k_2,
                       args.optimizer, args.batch_size)

    if args.dataset == "mosi":
        criterion = nn.L1Loss().cuda()
        t_in = 300
        a_in = 5
        v_in = 20
        label_dim = 1
        if args.aligned:
            T_t = T_a = T_v = 50
        else:
            T_t = 50
            T_a = 375
            T_v = 500

    elif args.dataset == "mosei_senti":
        criterion = nn.L1Loss().cuda()
        t_in = 300
        a_in = 74
        v_in = 35
        label_dim = 1
        if args.aligned:
            T_t = T_a = T_v = 50
        else:
            T_t, T_a, T_v = 50, 500, 500
    elif args.dataset == "iemocap":
        criterion = nn.CrossEntropyLoss().cuda()
        t_in = 300
        a_in = 74
        v_in = 35
        label_dim = 8
        if args.aligned:
            T_t = T_a = T_v = 20
        else:
            T_t, T_a, T_v = 20, 400, 500

    model = GCN_CAPS_Model(args, label_dim, t_in, a_in, v_in, T_t, T_a, T_v,
                           hyp_params.MULT_d,
                           hyp_params.vertex_num,
                           hyp_params.dim_capsule,
                           hyp_params.routing,
                           hyp_params.dropout).cuda()

    weight_decay = args.weight_decay
    if weight_decay > 0:
        reg_loss = Regularization(model, weight_decay, p=2).cuda()
    else:
        reg_loss = 0

    if args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    # 连续patience次验证集损失没有降低时降低学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, factor=0.1, verbose=True)

    train_data = get_data(args, args.dataset, 'train')
    valid_data = get_data(args, args.dataset, 'valid')
    test_data = get_data(args, args.dataset, 'test')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    if args.dataset in ['mosi', 'mosei_senti', 'iemocap']:
        best_valid = 1e9
        mae_best_acc = 2
        mult_a7_best_acc = -1
        mult_a5_best_acc = -1
        corr_best_acc = 0
        fscore_best_acc = 0
        patience_acc = 0

        # 保存每轮的效果用于绘图
        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []
        valid_loss_list = []
        valid_acc_list = []
        Neutral_f1_list = []
        Neutral_acc_list = []
        Happy_f1_list = []
        Happy_acc_list = []
        Sad_f1_list = []
        Sad_acc_list = []
        Angry_f1_list = []
        Angry_acc_list = []
        epoch_list = np.arange(1, args.epochs+1)

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        # adjust_k_2(epoch, args)
        # train for one epoch
        train_loss, train_results, train_truth = train(train_loader, model, criterion, optimizer, epoch,
                                                       weight_decay, reg_loss, args)
        # validate for one epoch
        valid_loss, valid_results, valid_truth = validate(valid_loader, model, criterion, args)
        # test for one epoch
        test_loss, test_results, test_truth = validate(test_loader, model, criterion, args)
        # 连续patience次验证集损失没有降低时降低学习率
        scheduler.step(valid_loss)

        if args.dataset == "mosi":
            mae_train, corr_train, mult_a7_train, mult_a5_train, f_score_train, acc_train = eval_mosi(train_results, train_truth)
            mae_test, corr_test, mult_a7_test, mult_a5_test, f_score_test, acc_test = eval_mosi(test_results, test_truth)
            mae_valid, corr_valid, mult_a7_valid, mult_a5_valid, f_score_valid, acc_valid = eval_mosi(valid_results, valid_truth)
        elif args.dataset == 'mosei_senti':
            mae_train, corr_train, mult_a7_train, mult_a5_train, f_score_train, acc_train = eval_mosei_senti(train_results, train_truth)
            mae_test, corr_test, mult_a7_test, mult_a5_test, f_score_test, acc_test = eval_mosei_senti(test_results, test_truth)
            mae_valid, corr_valid, mult_a7_valid, mult_a5_valid, f_score_valid, acc_valid = eval_mosei_senti(valid_results, valid_truth)
        elif args.dataset == 'iemocap':
            # Neutral_f1_train, Neutral_acc_train, Happy_f1_train, Happy_acc_train, Sad_f1_train, Sad_acc_train, Angry_f1_train, Angry_acc_train = eval_iemocap(train_results, train_truth)
            # Neutral_f1_val, Neutral_acc_val, Happy_f1_val, Happy_acc_val, Sad_f1_val, Sad_acc_val, Angry_f1_val, Angry_acc_val = eval_iemocap(valid_results, valid_truth)
            Neutral_f1_test, Neutral_acc_test, Happy_f1_test, Happy_acc_test, Sad_f1_test, Sad_acc_test, Angry_f1_test, Angry_acc_test = eval_iemocap(test_results, test_truth, False)
        # 记录每一轮的指标用于画图
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        valid_loss_list.append(valid_loss)
        # 显示每一轮的损失值并记录最好的结果
        if args.dataset in ['mosi', 'mosei_senti']:
            train_acc_list.append(acc_train)
            test_acc_list.append(acc_test)
            valid_acc_list.append(acc_valid)
            print('Epoch {:2d} Loss| Train Loss{:5.4f} | Valid Loss {:5.4f} | Test Loss {:5.4f} || Acc| Train Acc {:5.4f} | Valid Acc {:5.4f} | Test Acc {:5.4f}'
                  .format(epoch, train_loss, valid_loss, test_loss, acc_train, acc_valid, acc_test))
            print('Epoch {:2d} ACC| mae_test{:5.4f} | mult_a7_test {:5.4f} | corr_test {:5.4f} | f_score_test {:5.4f}'
                .format(epoch, mae_test, mult_a7_test, corr_test, f_score_test))
            # if acc_valid > best_valid:
            if valid_loss < best_valid:
                if args.aligned:
                    print('aligned {} dataset | acc improved! saving model to aligned_{}_best_model.pkl'
                          .format(args.dataset, args.dataset))
                    torch.save(model, 'aligned_{}_best_model.pkl'.format(args.dataset))
                else:
                    print('unaligned {} dataset | acc improved! saving model to unaligned_{}_best_model.pkl'
                          .format(args.dataset, args.dataset))
                    torch.save(model, 'unaligned_{}_best_model.pkl'.format(args.dataset))
                best_valid = valid_loss
                best_acc = acc_test
                mae_best_acc = mae_test
                mult_a7_best_acc = mult_a7_test
                mult_a5_best_acc = mult_a5_test
                corr_best_acc = corr_test
                fscore_best_acc = f_score_test
                patience_acc = 0
            else:
                patience_acc += 1
            # if patience_acc > 100:
            #     break
        elif args.dataset == 'iemocap':
            print('Epoch {:2d} Loss| Train Loss{:5.4f} | Valid Loss {:5.4f} | Test Loss {:5.4f} '
                .format(epoch, train_loss, valid_loss, test_loss))
            Neutral_f1_list.append(Neutral_f1_test)
            Neutral_acc_list.append(Neutral_acc_test)
            Happy_f1_list.append(Happy_f1_test)
            Happy_acc_list.append(Happy_acc_test)
            Sad_f1_list.append(Sad_f1_test)
            Sad_acc_list.append(Sad_acc_test)
            Angry_f1_list.append(Angry_f1_test)
            Angry_acc_list.append(Angry_acc_test)

    # 显示参数和最好的测试集结果
    if args.dataset in ['mosi', 'mosei_senti']:
        print("hyper-parameters: MULT_d, vertex_num, dim_capsule, routing, dropout, weight_decay, layers, k_1, k_2, "
              "optimizer, batch_size", current_setting)
        print("Best Acc: {:5.4f}".format(best_acc))
        print("mae: {:5.4f}".format(mae_best_acc))
        print("mult_a7: {:5.4f}".format(mult_a7_best_acc))
        print("mult_a5: {:5.4f}".format(mult_a5_best_acc))
        print("corr: {:5.4f}".format(corr_best_acc))
        print("fscore: {:5.4f}".format(fscore_best_acc))
        print('-' * 50)

        # 绘制折线图
        train_loss_list = np.around(train_loss_list, decimals=4)
        train_acc_list = np.around(train_acc_list, decimals=4)
        test_loss_list = np.around(test_loss_list, decimals=4)
        test_acc_list = np.around(test_acc_list, decimals=4)
        valid_loss_list = np.around(valid_loss_list, decimals=4)
        valid_acc_list = np.around(valid_acc_list, decimals=4)

        plt.figure(num=1)
        plt.plot(epoch_list, train_loss_list, "r", marker='D', markersize=1, label="train_loss")
        # plt.plot(epoch_list, test_loss_list, "g", marker='D', markersize=1, label="test_loss")
        # 绘制坐标轴标签
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Loss")
        plt.legend(loc="upper left")
        # 保存图片
        plt.savefig("train_loss.jpg")

        plt.figure(num=2)
        # plt.plot(epoch_list, train_loss_list, "r", marker='D', markersize=1, label="train_loss")
        plt.plot(epoch_list, valid_loss_list, "g", marker='D', markersize=1, label="valid_loss")
        # plt.plot(epoch_list, test_loss_list, "g", marker='D', markersize=1, label="test_loss")
        # 绘制坐标轴标签
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Loss")
        plt.legend(loc="upper left")
        # 保存图片
        plt.savefig("valid_loss.jpg")

        plt.figure(num=3)
        # plt.plot(epoch_list, train_loss_list, "r", marker='D', markersize=1, label="train_loss")
        # plt.plot(epoch_list, valid_loss_list, "r", marker='D', markersize=1, label="valid_loss")
        plt.plot(epoch_list, test_loss_list, "g", marker='D', markersize=1, label="test_loss")
        # 绘制坐标轴标签
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Loss")
        plt.legend(loc="upper left")
        # 保存图片
        plt.savefig("test_loss.jpg")

        plt.figure(num=4)
        # plt.plot(epoch_list, train_acc_list, label="train_acc_2")
        plt.plot(epoch_list, valid_acc_list, label="valid_acc_2")
        plt.xlabel("epochs")
        plt.ylabel("acc_2")
        plt.title("Acc_2")
        # for x1, y1 in zip(epoch_list, train_acc_list):
        #     plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        # for x1, y1 in zip(epoch_list, test_acc_list):
        #     plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        plt.legend(loc="upper left")
        plt.savefig("valid_acc_2.jpg")
        plt.show()

        plt.figure(num=5)
        # plt.plot(epoch_list, train_acc_list, label="train_acc_2")
        plt.plot(epoch_list, test_acc_list, label="test_acc_2")
        plt.xlabel("epochs")
        plt.ylabel("acc_2")
        plt.title("Acc_2")
        # for x1, y1 in zip(epoch_list, train_acc_list):
        #     plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        # for x1, y1 in zip(epoch_list, test_acc_list):
        #     plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        plt.legend(loc="upper left")
        plt.savefig("test_acc_2.jpg")
        plt.show()

    elif args.dataset == 'iemocap':
        train_loss_list = np.around(train_loss_list, decimals=4)
        test_loss_list = np.around(test_loss_list, decimals=4)
        Neutral_f1_list = np.around(Neutral_f1_list, decimals=4)
        Neutral_acc_list = np.around(Neutral_acc_list, decimals=4)
        Happy_f1_list = np.around(Happy_f1_list, decimals=4)
        Happy_acc_list = np.around(Happy_acc_list, decimals=4)
        Sad_f1_list = np.around(Sad_f1_list, decimals=4)
        Sad_acc_list = np.around(Sad_acc_list, decimals=4)
        Angry_f1_list = np.around(Angry_f1_list, decimals=4)
        Angry_acc_list = np.around(Angry_acc_list, decimals=4)

        plt.figure(num=1)
        plt.plot(epoch_list, train_loss_list, "r", marker='D', markersize=1, label="train_loss")
        plt.plot(epoch_list, test_loss_list, "g", marker='D', markersize=1, label="test_loss")
        # 绘制坐标轴标签
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Loss")
        # 调用 text()在图像上绘制注释文本
        # x1、y1表示文本所处坐标位置，ha参数控制水平对齐方式, va控制垂直对齐方式，str(y1)表示要绘制的文本
        for x1, y1 in zip(epoch_list, train_loss_list):
            plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        for x1, y1 in zip(epoch_list, test_loss_list):
            plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        plt.legend(loc="upper left")
        # 保存图片
        plt.savefig("loss.jpg")

        plt.figure(num=2)
        plt.plot(epoch_list, Neutral_f1_list, "r", marker='D', markersize=1, label="f1")
        plt.plot(epoch_list, Neutral_acc_list, "g", marker='D', markersize=1, label="acc")
        # 绘制坐标轴标签
        plt.xlabel("epochs")
        plt.ylabel("F1/Acc")
        plt.title("Neutral")
        # 调用 text()在图像上绘制注释文本
        # x1、y1表示文本所处坐标位置，ha参数控制水平对齐方式, va控制垂直对齐方式，str(y1)表示要绘制的文本
        for x1, y1 in zip(epoch_list, Neutral_f1_list):
            plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        for x1, y1 in zip(epoch_list, Neutral_acc_list):
            plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        plt.legend(loc="upper left")
        # 保存图片
        plt.savefig("Neutral.jpg")

        plt.figure(num=3)
        plt.plot(epoch_list, Happy_f1_list, "r", marker='D', markersize=1, label="f1")
        plt.plot(epoch_list, Happy_acc_list, "g", marker='D', markersize=1, label="acc")
        # 绘制坐标轴标签
        plt.xlabel("epochs")
        plt.ylabel("F1/Acc")
        plt.title("Happy")
        # 调用 text()在图像上绘制注释文本
        # x1、y1表示文本所处坐标位置，ha参数控制水平对齐方式, va控制垂直对齐方式，str(y1)表示要绘制的文本
        for x1, y1 in zip(epoch_list, Happy_f1_list):
            plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        for x1, y1 in zip(epoch_list, Happy_acc_list):
            plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        plt.legend(loc="upper left")
        # 保存图片
        plt.savefig("Happy.jpg")

        plt.figure(num=4)
        plt.plot(epoch_list, Sad_f1_list, "r", marker='D', markersize=1, label="f1")
        plt.plot(epoch_list, Sad_acc_list, "g", marker='D', markersize=1, label="acc")
        # 绘制坐标轴标签
        plt.xlabel("epochs")
        plt.ylabel("F1/Acc")
        plt.title("Sad")
        # 调用 text()在图像上绘制注释文本
        # x1、y1表示文本所处坐标位置，ha参数控制水平对齐方式, va控制垂直对齐方式，str(y1)表示要绘制的文本
        for x1, y1 in zip(epoch_list, Sad_f1_list):
            plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        for x1, y1 in zip(epoch_list, Sad_acc_list):
            plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        plt.legend(loc="upper left")
        # 保存图片
        plt.savefig("Sad.jpg")

        plt.figure(num=5)
        plt.plot(epoch_list, Angry_f1_list, "r", marker='D', markersize=1, label="f1")
        plt.plot(epoch_list, Angry_acc_list, "g", marker='D', markersize=1, label="acc")
        # 绘制坐标轴标签
        plt.xlabel("epochs")
        plt.ylabel("F1/Acc")
        plt.title("Angry")
        # 调用 text()在图像上绘制注释文本
        # x1、y1表示文本所处坐标位置，ha参数控制水平对齐方式, va控制垂直对齐方式，str(y1)表示要绘制的文本
        for x1, y1 in zip(epoch_list, Angry_f1_list):
            plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        for x1, y1 in zip(epoch_list, Angry_acc_list):
            plt.text(x1, y1, y1, ha='center', va='bottom', fontsize=10)
        plt.legend(loc="upper left")
        # 保存图片
        plt.savefig("Angry.jpg")
