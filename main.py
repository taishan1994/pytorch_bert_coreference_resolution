# coding=utf-8
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

import config
import dataset
import CRModel
from utils import utils
from utils import early_stop
from preprocess import CRBertFeature, get_data, CRProcessor

args = config.Args().get_parser()
utils.set_seed(args.seed)
logger = logging.getLogger(__name__)
utils.set_logger(os.path.join(args.log_dir, 'main.log'))


class BertForCR:
    def __init__(self, model, args):
        self.args = args
        self.model = model
        gpu_ids = args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.criterion = nn.CrossEntropyLoss()
        self.earlyStopping = early_stop.EarlyStopping(
            monitor='val_loss',
            patience=4,
            verbose=True,
            mode='min',
        )

    def build_optimizer_and_scheduler(self, t_total):
        module = (
            self.model.module if hasattr(model, "module") else self.model
        )

        # 差分学习率
        no_decay = ["bias", "LayerNorm.weight"]
        model_param = list(module.named_parameters())

        bert_param_optimizer = []
        other_param_optimizer = []

        for name, para in model_param:
            space = name.split('.')
            # print(name)
            if space[0] == 'bert_module':
                bert_param_optimizer.append((name, para))
            else:
                other_param_optimizer.append((name, para))

        optimizer_grouped_parameters = [
            # bert other module
            {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay, 'lr': self.args.lr},
            {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': self.args.lr},

            # 其他模块，差分学习率
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay, 'lr': self.args.other_lr},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': self.args.other_lr},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(self.args.warmup_proportion * t_total), num_training_steps=t_total
        )
        return optimizer, scheduler

    def train(self, train_loader, dev_loader=None):
        self.model.to(self.device)
        global_step = 0
        flag = False
        t_total = self.args.train_epochs * len(train_loader)
        eval_step = 10
        optimizer, scheduler = self.build_optimizer_and_scheduler(t_total)
        best_f1 = 0.0
        stop_count = 0
        stop_dev_loss = float('-inf')
        for epoch in range(1, self.args.train_epochs + 1):
            for step, batch_data in enumerate(train_loader):
                self.model.train()
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.device)
                output = self.model(
                    batch_data['token_ids'],
                    batch_data['attention_masks'],
                    batch_data['token_type_ids'],
                    batch_data['span1_ids'],
                    batch_data['span2_ids'],
                )
                loss = self.criterion(output, batch_data['label'])
                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                logger.info('[train] epoch:{}/{} step:{}/{} loss:{:.6f}'.format(epoch, self.args.train_epochs,
                                                                                global_step, t_total, loss.item()))
                global_step += 1
                if global_step % eval_step == 0:
                    dev_loss, accuracy, precision, recall, f1 = self.dev(dev_loader)
                    logger.info('[dev] loss:{:.6f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}'.format(
                        dev_loss, accuracy, precision, recall, f1))
                    self.earlyStopping(dev_loss, self.model)
                    if self.earlyStopping.early_stop:
                        flag = True
                        break
                    if f1 > best_f1:
                        best_f1 = f1
                        torch.save(self.model.state_dict(), self.args.output_dir + 'best.pt')
            if flag:
                break

    def dev(self, dev_loader):
        self.model.eval()
        self.model.to(self.device)
        total_loss = 0.0
        trues = []
        preds = []
        with torch.no_grad():
            for eval_step, dev_batch_data in enumerate(dev_loader):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(self.device)
                output = self.model(dev_batch_data['token_ids'],
                                    dev_batch_data['attention_masks'],
                                    dev_batch_data['token_type_ids'],
                                    dev_batch_data['span1_ids'],
                                    dev_batch_data['span2_ids'])
                label = dev_batch_data['label']
                loss = self.criterion(output, label)
                total_loss += loss.item()
                labels = label.cpu().detach().numpy().tolist()
                logits = np.argmax(output.cpu().detach().numpy().tolist(), -1)
                preds.extend(logits)
                trues.extend(labels)
            accuracy = accuracy_score(trues, preds)
            precision = precision_score(trues, preds, average='micro')
            recall = recall_score(trues, preds, average='micro')
            f1 = f1_score(trues, preds, average='micro')
            return total_loss, accuracy, precision, recall, f1

    def test(self, model, test_loader):
        model.eval()
        model.to(self.device)
        total_loss = 0.0
        trues = []
        preds = []
        with torch.no_grad():
            for eval_step, test_batch_data in enumerate(test_loader):
                for key in test_batch_data.keys():
                    test_batch_data[key] = test_batch_data[key].to(self.device)
                output = self.model(test_batch_data['token_ids'],
                                    test_batch_data['attention_masks'],
                                    test_batch_data['token_type_ids'],
                                    test_batch_data['span1_ids'],
                                    test_batch_data['span2_ids'])
                label = test_batch_data['label']
                loss = self.criterion(output, label)
                total_loss += loss.item()
                labels = label.cpu().detach().numpy().tolist()
                logits = np.argmax(output.cpu().detach().numpy().tolist(), -1)
                preds.extend(logits)
                trues.extend(labels)
            logger.info(classification_report(trues, preds))

    def predict(self, model, raw_text, span1, span2, args):
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            tokenizer = BertTokenizer(
                os.path.join(args.bert_dir, 'vocab.txt'))
            tokens = [i for i in raw_text]
            span1_ids = [0] * len(tokens)
            span1_start = span1[1]
            span1_end = span1_start + len(span1[0])

            for i in range(span1_start, span1_end):
                span1_ids[i] = 1
            span2_ids = [0] * len(tokens)
            span2_start = span2[1]
            span2_end = span2_start + len(span2[0])
            for i in range(span2_start, span2_end):
                span2_ids[i] = 1

            if len(span1_ids) <= args.max_seq_len - 2:  # 这里减2是[CLS]和[SEP]
                pad_length = args.max_seq_len - 2 - len(span1_ids)
                span1_ids = span1_ids + [0] * pad_length  # CLS SEP PAD label都为O
                span2_ids = span2_ids + [0] * pad_length
                span1_ids = [0] + span1_ids + [0]  # 增加[CLS]和[SEP]
                span2_ids = [0] + span2_ids + [0]
            else:
                if span2_end > max_seq_len - 2:
                    raise Exception('发生了不该有的截断')
                span1_ids = span1_ids[:args.max_seq_len - 2]
                span2_ids = span2_ids[:args.max_seq_len - 2]
                span1_ids = [0] + span1_ids + [0]  # 增加[CLS]和[SEP]
                span2_ids = [0] + span2_ids + [0]

            encode_dict = tokenizer.encode_plus(text=tokens,
                                                max_length=args.max_seq_len,
                                                padding="max_length",
                                                truncation='only_first',
                                                return_token_type_ids=True,
                                                return_attention_mask=True,
                                                return_tensors='pt', )
            token_ids = encode_dict['input_ids'].to(self.device)
            attention_masks = encode_dict['attention_mask'].to(self.device)
            token_type_ids = encode_dict['token_type_ids'].to(self.device)
            span1_ids = torch.tensor([span1_ids]).to(self.device)
            span2_ids = torch.tensor([span2_ids]).to(self.device)
            output = self.model(token_ids,
                                attention_masks,
                                token_type_ids,
                                span1_ids,
                                span2_ids)
            logits = np.argmax(output.cpu().detach().numpy().tolist(), -1)
            logger.info('结果：' + str(logits[0]))


if __name__ == '__main__':
    processor = CRProcessor()
    train_features = get_data(processor, 'train.json', 'train', args)
    train_dataset = dataset.CRDataset(train_features)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train_batch_size,
                              sampler=train_sampler,
                              num_workers=2)

    dev_features = get_data(processor, 'dev.json', 'dev', args)
    dev_dataset = dataset.CRDataset(dev_features)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.eval_batch_size,
                            num_workers=2)

    test_features = dev_features
    test_dataset = dataset.CRDataset(test_features)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.eval_batch_size,
                             num_workers=2)

    model = CRModel.CorefernceResolutionModel(args)
    bertForCR = BertForCR(model, args)

    # ===================================
    # bertForCR.train(train_loader, dev_loader)
    # ===================================
    # ===================================
    ckpt_path = './checkpoints/best.pt'
    model.load_state_dict(torch.load(ckpt_path))
    # bertForCR.test(model, test_loader)
    # ===================================
    # ===================================
    with open(args.data_dir + 'test.json', 'r') as fp:
        lines = fp.readlines()
        for i, line in enumerate(lines):
            data = eval(line)
            target = data['target']
            text = data['text']
            span1 = [target['span1_text'], target['span1_index']]
            span2 = [target['span2_text'], target['span2_index']]
            logger.info('===============================')
            logger.info('text=' + text)
            logger.info('span1=' + str(span1))
            logger.info('span2=' + str(span2))
            bertForCR.predict(model, text, span1, span2, args)
            logger.info('===============================')
            if i == 10:
                break
    # ===================================
