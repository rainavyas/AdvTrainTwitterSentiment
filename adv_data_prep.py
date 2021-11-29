'''
Load Adversarially Attacked Data from a Directory of subdirectories containing the adversarial examples

Expects a single text file per datapoint with the following structure:
    data_list: list
        [item1, item2, ...]
        where,
            item: dict
                with keys:
                    sentence
                    updated sentence
                    original prob
                    updated prob
                    true label
'''

import os
from os import walk
import json

def get_adv_example(filepath):
    '''
    Returns adversarial sentence and true label for a single file
    '''
    with open(filepath, 'r') as f:
        item = json.load(f)

    return item['updated sentence'], item['true label']


def get_adv(base_dir):
    
    subdirs = [ f.path for f in os.scandir(base_dir) if f.is_dir() ]
    tweets = []
    labels = []
    for dir in subdirs:
        filenames = next(walk(dir), (None, None, []))[2]
        for f in filenames:
            tweet, label = get_adv_example(dir+'/'+f)
            tweets.append(tweet)
            labels.append(label)
    return tweets, labels

