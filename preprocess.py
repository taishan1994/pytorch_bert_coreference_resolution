import os
import json
import logging
from transformers import BertTokenizer
from collections import defaultdict
from utils import utils
import config

logger = logging.getLogger(__name__)
args = config.Args().get_parser()
utils.set_seed(123)
# utils.set_logger(os.path.join(args.log_dir, 'preprocess.log'))

class InputExample:
    def __init__(self, set_type, text, span1=None, span2=None, label=None):
        self.set_type = set_type
        self.text = text
        self.span1 = span1
        self.span2 = span2
        self.label = label


class BaseFeature:
    def __init__(self, token_ids, attention_masks, token_type_ids):
        # BERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids


class CRBertFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 span1_ids=None,
                 span2_ids=None,
                 label=None):
        super(CRBertFeature, self).__init__(token_ids=token_ids,
                                            attention_masks=attention_masks,
                                            token_type_ids=token_type_ids)
        self.span1_ids = span1_ids
        self.span2_ids = span2_ids
        self.label = label


class CRProcessor:

    @staticmethod
    def read_json(file_path):
        with open(file_path, encoding='utf-8') as f:
            raw_examples = f.readlines()
        return raw_examples

    def get_examples(self, raw_examples, set_type):
        examples = []
        for raw_example in raw_examples:
            raw_example = eval(raw_example)
            text = raw_example['text']
            target = raw_example['target']
            span1 = [target['span1_text'], target['span1_index']]
            span2 = [target['span2_text'], target['span2_index']]
            label = 1 if raw_example['label'] == 'true' else 0
            examples.append(InputExample(set_type=set_type,
                                         text=text,
                                         span1=span1,
                                         span2=span2,
                                         label=label))
        return examples


def convert_rc_example(ex_idx, example: InputExample, tokenizer: BertTokenizer,
                        max_seq_len):
    set_type = example.set_type
    text = example.text
    span1 = example.span1
    span2 = example.span2
    label = example.label
    features = []

    tokens = [i for i in text]
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

    if len(span1_ids) <= max_seq_len - 2: # 这里减2是[CLS]和[SEP]
        pad_length = max_seq_len - 2 - len(span1_ids)
        span1_ids = span1_ids + [0] * pad_length  # CLS SEP PAD label都为O
        span2_ids = span2_ids + [0] * pad_length
        span1_ids = [0] + span1_ids + [0]  # 增加[CLS]和[SEP]
        span2_ids = [0] + span2_ids + [0]
    else:
        if span2_end > max_seq_len - 2:
            return '发生了不该有的截断'
        span1_ids = span1_ids[:max_seq_len - 2]
        span2_ids = span2_ids[:max_seq_len - 2]
        span1_ids = [0] + span1_ids + [0]  # 增加[CLS]和[SEP]
        span2_ids = [0] + span2_ids + [0]

    assert len(span1_ids) == max_seq_len
    assert len(span2_ids) == max_seq_len

    # 随机mask
    # if mask_prob:
    #     tokens_b = sent_mask(tokens_b, stop_mask_ranges, mask_prob=mask_prob)

    encode_dict = tokenizer.encode_plus(text=tokens,
                                        max_length=max_seq_len,
                                        padding="max_length",
                                        truncation='only_first',
                                        return_token_type_ids=True,
                                        return_attention_mask=True)

    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    if ex_idx < 3:
        logger.info(f"*** {set_type}_example-{ex_idx} ***")
        logger.info(f'text {tokens}')
        logger.info(f"token_ids: {token_ids}")
        logger.info(f"attention_masks: {attention_masks}")
        logger.info(f"token_type_ids: {token_type_ids}")
        logger.info(f'span1_ids: {span1_ids}')
        logger.info(f"span2_ids: {span2_ids}")
        logger.info(f"label: {label}")

    feature = CRBertFeature(token_ids=token_ids,
                             attention_masks=attention_masks,
                             token_type_ids=token_type_ids,
                             span1_ids=span1_ids,
                             span2_ids=span2_ids,
                             label=label,
                             )

    features.append(feature)

    return features


def convert_examples_to_features(examples, max_seq_len, bert_dir):
    tokenizer = BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))
    features = []
    logger.info(f'Convert {len(examples)} examples to features')

    for i, example in enumerate(examples):
        feature = convert_rc_example(
            ex_idx=i,
            example=example,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
        )
        if feature is None or feature == '发生了不该有的截断':
            continue
        features.extend(feature)

    logger.info(f'Build {len(features)} features')
    return features


def get_data(processor, json_file, mode, args):
    raw_examples = processor.read_json(os.path.join(args.data_dir, json_file))
    examples = processor.get_examples(raw_examples, mode)
    data = convert_examples_to_features(examples, args.max_seq_len, args.bert_dir)
    return data

if __name__ == '__main__':
    processor = CRProcessor()

    train_data = get_data(processor, "train.json", "train", args)
    dev_data = get_data(processor, "dev.json", "dev", args)
    test_data = get_data(processor, "dev.json", "test", args)
