"""
nlidataset.py

Part of NLI script in gluon-nlp.
Copyright 2018 Mengxiao Lin <linmx0130@gmail.com>
"""

import json
import mxnet.gluon.data as gdata

class NLIDataItem:
    """
    Natural Language Inference data item class.
    Simply acces `gold_label`, `sentence1` and `sentence2`
    """
    gold_label = None
    sentence1 = None
    sentence2 = None

    def __init__(self):
        pass
    def parse_tab_line(self, line: str):
        """
        Parse the tab format of NLI datasets.
        """
        fields = line.strip().split('\t')
        self.gold_label = fields[0]
        self.sentence1 = fields[5]
        self.sentence2 = fields[6]
    def parse_json_line(self, line: str):
        """
        Parse JSON format of NLI datasets.
        """
        data_item = json.loads(line)
        self.gold_label = data_item['gold_label']
        self.sentence1 = data_item['sentence1']
        self.sentence2 = data_item['sentence2']


class NLIDataset(gdata.SimpleDataset):
    """
    Dataset for NLI.
    """
    def __init__(self, filename, parse_type='tab'):
        """
        Arguments:
            filename: the input filename
            parse_type: 'tab' or 'json'
        """
        self.parse_type = parse_type
        assert parse_type in ['tab', 'json']
        data = self._read_data(filename)
        super(NLIDataset, self).__init__(data)

    def _read_data(self, filename):
        with open(filename) as handler:
            raw_data = handler.readlines()
        data = []
        if self.parse_type == 'tab':
            for line in raw_data[1:]:
                item = NLIDataItem()
                item.parse_tab_line(line)
                data.append(item)
        if self.parse_type == 'json':
            for line in raw_data[1:]:
                item = NLIDataItem()
                item.parse_json_line(line)
                data.append(item)
        return data
