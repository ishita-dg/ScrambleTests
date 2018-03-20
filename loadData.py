"""
Terminology:

A *dataset* is of the form {
  "X_A": [sentence1a, sentence2a, ...],
  "X_B": [sentenec1b, sentence2b, ...],
  "y": [label1, label2, ...]
}

A *data group* is of the form {
  "train": dataset1,
  "test": dataset2,
  "dev": dataset3
}

"""

import numpy as np
import os

from IPython import embed


def load_lines(path):
    out = []
    with open(path, 'r') as f:
        for line in f:
            text = line.strip()
            out.append(text)
    return out


def load_labels(path, label2id):
    out = []
    with open(path, 'r') as f:
        for line in f:
            text = line.strip()
            out.append(label2id[text.upper()])
    return out


def sort_dataset(dataset):
    """
    Sort to reduce padding. Doesn't mutate original dataset.
    """
    tuples = sorted(zip(dataset['X_A'], dataset['X_B'], dataset['y']),
                    key=lambda z:(len(z[0]), len(z[1]), z[2]))
    sorted_dataset = {
        'X_A': [x for (x,y,z) in tuples],
        'X_B': [y for (x,y,z) in tuples],
        'y': [z for (x,y,z) in tuples]
    }    
    return sorted_dataset


def sort_group(group):
    return {k: sort_dataset(v) for k, v in group.items()}


def loadSNLI(path, label2id):
    """
    Returns SNLI data group
    """
    snli_data = {'train':{}, 'dev':{}, 'test':{}}

    for key in snli_data:  
        print('Loading data for {0}'.format(key))
        snli_data[key] = {
            'X_A': load_lines(path + 's1.' + key),
            'X_B': load_lines(path + 's2.' + key), 
            'y': load_labels(path + 'labels.' + key, label2id)
        }
        snli_data[key] = sort_dataset(snli_data[key])
    
    return snli_data


def init_dataset():
    return {
        "X_A": [],
        "X_B": [],
        "y": []
    }

def init_group():
    return {
      "train": init_dataset(),
      "dev": init_dataset(),
      "test": init_dataset()
    }


def merge_groups(groups):
    """
    Merge multiple groups into a single one
    """
    merged_group = init_group()
    for group in groups:
        for dataset_type in merged_group.keys():
            for k in merged_group[dataset_type].keys():  # k in ("X_A", "X_B", "y")
                merged_group[dataset_type][k] += group[dataset_type][k]
    return merged_group


def load_scramble_task(path, label2id, task, one_group=False):
    """
    Return data group for a single scramble task. Not sorted yet.
    """
    scramble_group = init_group()
    dataset_types = list(scramble_group.keys())
    
    label_path = os.path.join(path, "labels.{0}".format(task))
    s1_path = os.path.join(path, "s1.{0}".format(task))
    s2_path = os.path.join(path, "s2.{0}".format(task))
    
    with open(s1_path) as s1_file, open(s2_path) as s2_file, open(label_path) as label_file:
         for i, (s1, s2, label) in enumerate(zip(s1_file, s2_file, label_file)):
             group_type = dataset_types[i % 3] if not one_group else "train"
             dataset = scramble_group[group_type]
             dataset['X_A'].append(s1.strip())
             dataset['X_B'].append(s2.strip())
             dataset['y'].append(int(label.strip()))
    #embed()
    return scramble_group


def load_scramble_all(path, label2id, tasks, one_group=False):
    """
    Load all scramble data groups merged together as a single group, sorted to reduce padding.
    """
    groups = []
    for task in tasks:
        groups.append(load_scramble_task(path, label2id, task, one_group))
    big_group = merge_groups(groups)
    if one_group:
        print("load scramble all")
        #embed()
        return big_group["train"]
    sorted_group = sort_group(big_group)
    return sorted_group

