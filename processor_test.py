class CRProcessor:

    def get_result(self, path):
        with open(path,'r') as fp:
            data = fp.readlines()
        for d in data:
            d = eval(d)
            target = d['target']
            text = d['text']
            span1 = [target['span1_text'], target['span1_index']]
            span2 = [target['span2_text'], target['span2_index']]
            label = 1 if d['label'] == 'True' else 0
            print(text, span1, span2, label)
            break

if __name__ == '__main__':
    crProcessor = CRProcessor()
    crProcessor.get_result('../../data/coreference_resolution/train.json')