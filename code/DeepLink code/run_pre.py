from __future__ import absolute_import, division, print_function

import datetime
import gc
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, \
    roc_auc_score, matthews_corrcoef, brier_score_loss, confusion_matrix
from torch.utils.data import DataLoader, \
    SequentialSampler, RandomSampler
from tqdm import tqdm
from transformers import (AdamW, get_linear_schedule_with_warmup)

# from model2 import BTModel
from utils import getargs, set_seed, TextDataset

warnings.filterwarnings(action='ignore')

best_f = -1.0
inline_dataset_best_f = -1.0
test_dataset = "B"


def train(args, model, train_dataset, eval_dataset):
    dfScores = pd.DataFrame(columns=['Epoch', 'Metrics', 'Score'])
    torch.set_grad_enabled(True)
    # Prepare optimizer and schedule (linear warmup and decay)
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=4, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    max_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1,
                                                num_training_steps=max_steps)

    # Train!
    print("********** Running training **********")
    print("  Num examples = {}".format(len(train_dataset)))
    print("  Num Epochs = {}".format(args.num_train_epochs))
    print("  batch size = {}".format(args.train_batch_size))
    print("  Total optimization steps = {}".format(max_steps))
    global best_f, inline_dataset_best_f
    model.zero_grad()

    model.train()
    for idx in range(args.num_train_epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        losses = []

        for step, batch in enumerate(bar):
            loss, logits, = model(batch[0].to(args.device),
                                  batch[1].to(args.device),
                                  batch[2].to(args.device),
                                  batch[3].to(args.device), )
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            losses.append(loss.item())
            bar.set_description(
                "epoch {} loss {}".format(idx, round(float(np.mean(losses)), 3)))

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            del loss
            del logits
            torch.cuda.empty_cache()
            gc.collect()  # 清理内存
        results = evaluate(args, model, eval_dataset)
        for key, value in results.items():
            print('-' * 10 + "  {} = {}".format(key, round(value, 4)))
        for key in sorted(results.keys()):
            print('-' * 10 + "  {} = {}".format(key, str(round(results[key], 4))))
            dfScores.loc[len(dfScores)] = [idx, key, str(round(results[key], 4))]

        if results['eval_f1'] > best_f:
            inline_dataset_best_f = results['eval_f1']
            best_f = results['eval_f1']
            print("  " + "*" * 20)
            print("  Best f1: {}".format(round(best_f, 4)))
            print("  " + "*" * 20)
            checkpoint_prefix = args.pro + '_checkpoint-best'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
            torch.save(model, output_dir)
            print("Saving model checkpoint to {}".format(output_dir))
            dfScores.loc[len(dfScores)] = [idx, '___best___', '___best___']
        elif results['eval_f1'] > inline_dataset_best_f:
            inline_dataset_best_f = results['eval_f1']
            print("  " + "*" * 20)
            print("  Inline Dataset Best f1: {}".format(round(inline_dataset_best_f, 4)))
            print("  " + "*" * 20)
            checkpoint_prefix = args.pro + '_checkpoint-best'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
            torch.save(model, output_dir)
            print("Saving model checkpoint to {}".format(output_dir))
            dfScores.loc[len(dfScores)] = [idx, '___inline___', '___inline___']
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        dfScores.to_csv(os.path.join(args.result_dir, args.pro + "Epoch_Metrics.csv"), index=False)
        model.train()


def evaluate(args, model, eval_dataset):
    stime = datetime.datetime.now()
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    # 顺序采样
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4,
                                 pin_memory=True)

    # Eval!
    print("***** Running evaluation *****")
    print("  Num examples = {}".format(len(eval_dataset)))
    print("  Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in eval_dataloader:
        label = batch[3].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(batch[0].to(args.device),
                                   batch[1].to(args.device),
                                   batch[2].to(args.device),
                                   label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits.argmax(-1)
    print('Predictions', preds[:25])
    print('Labels:', labels[:25])
    etime = datetime.datetime.now()
    eval_time = (etime - stime).seconds
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    eval_precision = precision_score(labels, preds)
    eval_recall = recall_score(labels, preds)
    eval_f1 = f1_score(labels, preds)
    eval_auc = roc_auc_score(labels, preds)
    eval_mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=preds).ravel()
    eval_pf = fp / (fp + tn)
    eval_brier = brier_score_loss(labels, preds)

    result = {
        "eval_loss": float(perplexity),
        "eval_time": float(eval_time),
        "eval_acc": round(float(eval_acc), 4),
        "eval_precision": round(eval_precision, 4),
        "eval_recall": round(eval_recall, 4),
        "eval_f1": round(eval_f1, 4),
        "eval_auc": round(eval_auc, 4),
        "eval_mcc": round(eval_mcc, 4),
        "eval_brier": round(eval_brier, 4),
        "eval_pf": round(eval_pf, 4),
    }
    return result


def model_test(args, test_dataset, model, last=""):
    # Note that DistributedSampler samples randomly
    stime = datetime.datetime.now()
    eval_dataset = TextDataset(args, os.path.join(args.data_dir, test_dataset + '_TEST.csv'))
    args.seed += 3
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    print("********** Running Test **********")
    print("  Num examples = {}".format(len(eval_dataset)))
    print("  Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    issues = []
    commits = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        label = batch[3].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device), label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
            for iss, com in zip(batch[4], batch[5]):
                issues.append(iss)
                commits.append(com)
        nb_eval_steps += 1

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits.argmax(-1)
    etime = datetime.datetime.now()

    eval_time = (etime - stime).seconds
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    eval_precision = precision_score(labels, preds)
    eval_recall = recall_score(labels, preds)
    eval_f1 = f1_score(labels, preds)
    eval_auc = roc_auc_score(labels, preds)
    eval_mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=preds).ravel()
    eval_pf = fp / (fp + tn)
    eval_brier = brier_score_loss(labels, preds)

    result = {
        "eval_loss": float(perplexity),
        "eval_time": float(eval_time),
        "eval_acc": round(float(eval_acc), 4),
        "eval_precision": round(eval_precision, 4),
        "eval_recall": round(eval_recall, 4),
        "eval_f1": round(eval_f1, 4),
        "eval_auc": round(eval_auc, 4),
        "eval_mcc": round(eval_mcc, 4),
        "eval_brier": round(eval_brier, 4),
        "eval_pf": round(eval_pf, 4),
    }
    print(preds[:25], labels[:25])
    print("********** Test results **********")
    dfScores = pd.DataFrame(columns=['Metrics', 'Score'])
    for key in sorted(result.keys()):
        print('-' * 10 + "  {} = {}".format(key, str(round(result[key], 4))))
        dfScores.loc[len(dfScores)] = [key, str(round(result[key], 4))]
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    dfScores.to_csv(os.path.join(args.result_dir, args.pro + "_Metrics_" + last + ".csv"), index=False)
    assert len(logits) == len(preds) and len(logits) == len(labels), 'error'
    return round(eval_f1, 4)


key_dict = {
    "A": "#",
    "B": "LOG4NET",
    "C": "GIRAPH",
    "D": "OODT",
    "E": "NUTCH"
}


def main():
    print("======BTLink BEGIN...======" * 5)
    args = getargs()
    print(args.key)
    print("device: {}, n_gpu: {}".format(args.device, args.n_gpu))
    for i in range(args.n_gpu):
        device = torch.device(f"cuda:{i}")
        properties = torch.cuda.get_device_properties(device)
        print(f"GPU {i} information: ")
        print("name: {}, memory size: {}".format(properties.name, properties.total_memory))
    # Set seed
    set_seed(args.seed)

    global inline_dataset_best_f, best_model, best_f  # 表示在当前数据集下的最好的分数记录
    test_dataset = args.mode[-1]
    best_model = args.mode[0]
    for i in range(len(args.mode) - 1):
        inline_dataset_best_f = -1.0  # 第一次训练这个数据集要初始化
        args.pro = args.mode[:i + 2]
        print(f'===Project:{args.pro}---{args.pro}===' * 2)
        # 判断是否训练过，如果已经有了就不需要训练了
        checkpoint_prefix = args.pro[:i + 2] + '_checkpoint-best/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        if os.path.exists(output_dir):
            print(args.pro + "数据集以及存在了")
            model = torch.load(output_dir)
            model.to(args.device)
            f1 = model_test(args, test_dataset, model)
            if f1 > best_f:
                best_model = args.pro[:i + 2]
                best_f = f1
            continue
        # 判断读哪个模型
        checkpoint_prefix = best_model + '_checkpoint-best/model.bin'
        # 读取模型
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model = torch.load(output_dir)
        model.to(args.device)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        print("Training/evaluation parameters {}".format(args))

        if len(args.pro) == 5:
            print("训练无数据集的效果")
            model_test(args, test_dataset, model, last="last")

        # Training
        if args.do_train:
            args.seed += 3
            # load dataset
            args.key = key_dict.get(args.pro[-1])
            train_dataset = TextDataset(args, os.path.join(args.data_dir, args.pro[-1] + '_TRAIN.csv'))
            eval_dataset = TextDataset(args, os.path.join(args.data_dir, test_dataset + '_TEST.csv'))
            train(args, model, train_dataset, eval_dataset)


if __name__ == "__main__":
    main()
