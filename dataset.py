import torch
from torch.utils.data import Dataset, DataLoader


class CRDataset(Dataset):
    def __init__(self, features):
        self.nums = len(features)
        self.features = features

    def __len__(self):
        return self.nums

    def __str__(self):
        return self.__dict__

    def __getitem__(self, index):
        data = {
            'token_ids': torch.tensor(self.features[index].token_ids).long(),
            'attention_masks': torch.tensor(self.features[index].attention_masks, dtype=torch.uint8),
            'token_type_ids': torch.tensor(self.features[index].token_type_ids).long()
        }

        data['span1_ids'] = torch.tensor(self.features[index].span1_ids).long()
        data['span2_ids'] = torch.tensor(self.features[index].span2_ids).long()
        data['label'] = torch.tensor(self.features[index].label).long()

        return data


if __name__ == '__main__':
    import config
    # 这里要显示的引入CRBertFeature，不然会报错
    from preprocess import CRBertFeature, CRProcessor, get_data
    args = config.Args().get_parser()
    processor = CRProcessor()
    train_data = get_data(processor, "train.json", "train", args)
    train_features = train_data
    for train_feature in train_features:
        print(train_feature.token_ids)
        print(train_feature.attention_masks)
        print(train_feature.token_type_ids)
        print(train_feature.span1_ids)
        print(train_feature.span2_ids)
        break

    crDataset = CRDataset(train_features)
    print(crDataset[0])
