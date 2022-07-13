from matplotlib.cbook import maxdict
import numpy as np
import json
import xml.etree.cElementTree as ET
from ete3 import Tree, TreeStyle
import os
import copy
import itertools
import matplotlib.pyplot as plt

class Node:
    def __init__(self, id=None, name=None) -> None:
        self.id = id
        self.name = name
        self.parent = None
        self.children = []
        self.leaves = []

def expand_tree(node, max_d):
    if len(node.children)==0 and node.depth<max_d:
        par = node.parent
        idx = par.children.index(node)
        d = node.depth
        name = node.name
        cur = node
        while d < max_d:
            inter_node = Node(name=name+'_'+str(d))
            inter_node.children = [cur]
            cur = inter_node
            d += 1
        par.children[idx] = cur
    else:
        for c in node.children:
            expand_tree(c, max_d)

def tree_to_json(node):
    js = {}
    js['name'] = node.name
    # js['class_name'] = node.class_name
    if len(node.children)>0:
        js['children'] = []
        for c in node.children:
            js['children'].append(tree_to_json(c))
    return js

def json_to_tree(js):
    node = Node(name=js['name'])
    # node.class_name = js['class_name']
    if 'children' in js:
        for c in js['children']:
            c_tree = json_to_tree(c)
            c_tree.parent = node
            node.children.append(c_tree)
    return node


def get_max_depth(node, d):
    if node.depth>d[0]:
        d[0]=node.depth
    for c in node.children:
        get_max_depth(c, d)

def tree_to_c_hierarchy(tree):
    """data structure for c
    """
    hierarchy = []
    def travel(node):
        if len(node.children)>0:
            hierarchy.append({
                node.name: [c.name for c in node.children]
                })
            for c in node.children:
                travel(c)
    travel(tree)
    return hierarchy


def hierarchy_json_to_txt(json_path, txt_path):
    js = json.load(open(json_path, 'r', encoding="utf-8"))
    tree = json_to_tree(js)
    hierarchy = tree_to_c_hierarchy(tree)
    hierarchy_file = open(txt_path, mode='w')
    for h in hierarchy:
        k = list(h.items())[0][0]
        v = list(h.items())[0][1]
        hierarchy_file.write(k)
        hierarchy_file.write(' ')
        for item in v:
            hierarchy_file.write(item)
            hierarchy_file.write(' ')
        hierarchy_file.write('\n')
    hierarchy_file.close()

def name2id_npy_to_txt(npy_path, txt_path):
    name2id = np.load(npy_path, allow_pickle=True).item()

    name2id_file = open(txt_path, mode='w')
    for k, v in name2id.items():
        name2id_file.write(k)
        name2id_file.write(' ')
        name2id_file.write(str(v))
        name2id_file.write('\n')
    name2id_file.close()

def write_case(n, A, B, path):
    f = open(path, mode='w')
    f.write(str(n))
    f.write('\n')
    f.write('\n')
    for i in range(n):
        for j in range(n):
            f.write(str(int(A[i][j]))+' ')
        f.write('\n')
    f.write('\n')
    for i in range(n):
        for j in range(n):
            f.write(str(int(B[i][j]))+' ')
        f.write('\n')  
    f.write('\n')  