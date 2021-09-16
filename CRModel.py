import torch
import torch.nn as nn
from bertBaseModel import BaseModel


class CorefernceResolutionModel(BaseModel):
    def __init__(self,
                 args,
                 **kwargs):
        """
        tag the subject and object corresponding to the predicate
        :param use_type_embed: type embedding for the sentence
        :param loss_type: train loss type in ['ce', 'ls_ce', 'focal']
        """
        super(CorefernceResolutionModel, self).__init__(args.bert_dir, dropout_prob=args.dropout_prob)

        out_dims = self.bert_config.hidden_size
        mid_linear_dims = kwargs.pop('mid_linear_dims', 128)

        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        out_dims = mid_linear_dims

        self.fc = nn.Linear(out_dims, 2)
        self.dropout = nn.Dropout(0.3)
        init_blocks = [self.mid_linear, self.fc]

        self._init_weights(init_blocks)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                span1_mask=None,
                span2_mask=None):
        bert_outputs = self.bert_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        token_out = bert_outputs[0] # [batch, max_seq_len, dim]
        seq_out = bert_outputs[1]  # [batch, dim]
        logits = []
        for t_out, s_out, s1_mask, s2_mask in zip(token_out, seq_out, span1_mask, span2_mask):
            s1_mask = s1_mask == 1
            s2_mask = s2_mask == 1
            span1_out = t_out[s1_mask]
            span2_out = t_out[s2_mask]
            out = torch.cat([s_out.unsqueeze(0), span1_out, span2_out], dim=0).unsqueeze(0)
            # 这里可以使用最大池化或者平均池化，使用平均池化的时候要注意，
            # 要除以每一个句子本身的长度
            out = torch.sum(out, 1)
            logits.append(out)
        logits = torch.cat(logits, dim=0)
        logits = self.mid_linear(logits)
        logits = self.dropout(logits)
        logits = self.fc(logits)
        return logits



if __name__ == '__main__':
    class Args:
        bert_dir = '../../model_hub/hfl_chinese-roberta-wwm-ext/'
        dropout_prob = 0.3
    args = Args()
    model = CorefernceResolutionModel(args)
    print(model)
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_dir)
    sentence = '塑料椅子这边坐着很多候烧者，沙发那边只有五个候烧者，他们舒适地架着二郎腿，都是一副功成名就的>模样，塑料椅子这边的个个都是正襟危坐。'
    inputs = tokenizer.encode_plus(sentence,
                                   return_attention_mask=True,
                                   return_token_type_ids=True,
                                   return_tensors='pt')
    outputs = model(**inputs)

