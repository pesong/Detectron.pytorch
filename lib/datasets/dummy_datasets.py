# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
"""Provide stub objects that can act as stand-in "dummy" datasets for simple use
cases, like getting all classes in a dataset. This exists so that demos can be
run without requiring users to download/install datasets first.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import yaml

from utils.collections import AttrDict
from datasets.dataset_catalog import ANN_FN
from datasets.dataset_catalog import DATASETS
import json



def get_dataset(dataset):
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = []


    ann_json_path = DATASETS[dataset][ANN_FN]

    with open(ann_json_path) as f:
        line = f.readline()
        d = json.loads(line)
        categories = d['categories']

    for cat in categories:
        classes.append(cat['name'])

    classes.insert(0, '__background__')

    # classes = [
    #     '__background__',
    #     'road',
    #     'car',
    #     'truck',
    #     'bus',
    #     'bicycle',
    #     'motorcycle',
    #     'rider',
    #     'person'
    # ]

    print(classes)
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds


if __name__ == '__main__':
    get_dataset('cityscapes_val')
