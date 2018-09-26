#!/usr/bin/env python
# nlidataset.py
# Copyright 2018 Mengxiao Lin <linmx0130@gmail.com>
# 

import mxnet.gluon.data as gdata
import json

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
        
    def parse_tab_line(self, line:str):
        fields = line.strip().split("\t")
        self.gold_label = fields[0]
        self.sentence1 = fields[5]
        self.sentence2 = fields[6]
    
    def parse_json_line(self, line:str):
        data_item = json.loads(line)
        self.gold_label = data_item['gold_label']
        self.sentence1 = data_item['sentence1']
        self.sentence2 = data_item['sentence2']


class NLIDataset(gdata.SimpleDataset):
    """
    """
    def __init__(self, filename, parse_type="tab"):
        """
        Arguments:
            filename: the input filename
            parse_type: "tab" or "json"
        """
        self.parse_type = parse_type
        assert parse_type in ["tab", "json"]
        data = self._read_data(filename)
        super(NLIDataset, self).__init__(data)

    def _read_data(self, filename):
        with open(filename) as f:
            raw_data = f.readlines()
        data = []
        if self.parse_type == "tab":
            for l in raw_data[1:]:
                item = NLIDataItem()
                item.parse_tab_line(l)
                data.append(item)
        if self.parse_type == "json":
            for l in raw_data[1:]:
                item = NLIDataItem()
                item.parse_json_line(l)
                data.append(item)
        return data
    
