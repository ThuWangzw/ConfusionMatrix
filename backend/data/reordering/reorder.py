from collections import deque
import numpy as np
from .tree import TreeNode
# from .pkgR import seriation
from .measure import Measure
import copy
import matplotlib.pyplot as plt
import time
import datetime
import itertools
import os
import math
import os.path as osp
import json
import pandas as pd


INTERVAL_NODE_START_ID = 2000
MAX_DIS = 1000000

"""
    Reordering Algorithms
"""

def statis():

    # test_on_dataset(gamma=1, use_level=False, num_iters=50000, preorder=False, restart=5, dir_name='organism')
    # test_on_dataset(gamma=1, use_level=True, num_iters=50000, preorder=False, restart=1, dir_name='organism')
    # test_on_dataset(gamma=1, use_level=False, num_iters=10000, preorder=True, restart=5, dir_name='organism')
    # test_on_dataset(gamma=1, use_level=False, num_iters=2000, preorder=True, restart=5, dir_name='organism')
    # test_on_dataset(gamma=1, use_level=False, num_iters=50000, preorder=False, restart=5, dir_name='all')
    # test_on_dataset(gamma=1, use_level=True, num_iters=50000, preorder=False, restart=1, dir_name='all')
    # test_on_dataset(gamma=1, use_level=False, num_iters=10000, preorder=True, restart=5, dir_name='all')
    # test_on_dataset(gamma=1, use_level=False, num_iters=4000, preorder=True, restart=5, dir_name='all')

    # test_on_dataset(gamma=1, use_level=False, num_iters=10000, preorder=False, restart=5, dir_name='100')
    # test_on_dataset(gamma=1, use_level=True, num_iters=10000, preorder=False, restart=5, dir_name='100')
    # test_on_dataset(gamma=1, use_level=False, num_iters=1000, preorder=True, restart=5, dir_name='100')

    # test_on_dataset(gamma=1, use_level=False, num_iters=10000, preorder=False, restart=5, dir_name='80')
    # test_on_dataset(gamma=1, use_level=True, num_iters=10000, preorder=False, restart=5, dir_name='80')
    test_on_dataset(gamma=1, use_level=False, num_iters=1000, preorder=True, restart=5, dir_name='80')

    print()
    
def test_on_dataset(gamma=1, num_iters=35000, use_level=False, preorder=False, restart = 5, dir_name = 'aircraft'):
    
    # avgs = [0, 170827915.1, 36788362615.5, 1931046872871.8, 12153108876535.5, 20005951465490.1]
    # stds = [0, 304550.4, 1156040.0, 1893197.5, 3094937.7, 4303701.3]

    from .data.imagenet.read_hierarchy import get_tree, get_name2id, write_case
    from .pkgC.SA.agg_sa import getQAPReordering_file, getQAPReordering, getAvgAndStd

    
    path = '/data/fengyuan/72/ConfusionMatrix/backend/data/reordering/data/coco/' + dir_name

    suffix = dir_name
    suffix += f'_iters_{num_iters}_restart_{restart}'
    if use_level: suffix += '_level'
    if preorder: suffix += '_pre'

    hierarchy_json_path = osp.join(path, 'hierarchy.json')
    hierarchy_txt_path = osp.join(path, 'hierarchy.txt')
    hierarchy_pre_json_path = osp.join(path, 'hierarchy_pre.json')
    hierarchy_pre_txt_path = osp.join(path, 'hierarchy_pre.txt')
    name2id_npy_path = osp.join(path, 'name2id.npy')
    name2id_txt_path = osp.join(path, 'name2id.txt')
    matrix_txt_path = osp.join(path, 'matrix.txt')
    matrix_npy_path = osp.join(path, 'conf_mat.npy')
    avg_and_std_path = osp.join(path, 'avg_std.npy')

    if not osp.exists(avg_and_std_path):
        avg_std = getAvgAndStd(True, matrix_txt_path, hierarchy_txt_path, name2id_txt_path, 5000)
        avgs = [0] + avg_std[0][::-1]
        stds = [0] + avg_std[1][::-1]
        avg_std = [avgs, stds]
        np.save(avg_and_std_path, avg_std)
    avg_std = np.load(avg_and_std_path, allow_pickle=True)
    avgs = avg_std[0]
    stds = avg_std[1]

    tmp_dic_path = osp.join(path, "tmp_dic.npy") 
    tmp_matrix_txt_path = osp.join(path, 'tmp_matrix.txt')
    if not osp.exists(osp.join(path, 'results')):
        os.mkdir(osp.join(path, 'results'))
    results_dic_path = osp.join(path, 'results', 'results_dic.npy')
    results_csv_path = osp.join(path, 'results', 'results_dic.csv')

    if preorder:
        hierarchy_path = hierarchy_pre_txt_path
        tree = get_tree(hierarchy_pre_json_path)
    else:
        hierarchy_path = hierarchy_txt_path
        tree = get_tree(hierarchy_json_path)

    name2id = get_name2id(name2id_npy_path)
    CM = np.load(matrix_npy_path, allow_pickle=True)
    setIds(tree, name2id)
    setDepth(tree, 0)
    setLeaves(tree)

    confusion_dic = {}
    getConfusionDic(tree, CM, confusion_dic)

    num_perm = calcPermutations(tree)
    # print('num_perm', num_perm)


    def draw(M, perm, img_path):
        M = np.array(M)
        if perm:
            M = M[perm][:, perm]
        A = M.tolist()
        ax = plt.matshow(A)
        plt.colorbar(ax.colorbar, fraction=0.025)
        plt.title(img_path)
        plt.savefig(img_path)
        plt.cla()
        plt.close()

    def draw_pair(M, perm, name):
        M = np.array(M)
        p_M = np.array(M)
        p_M = p_M[perm][:, perm]

        fig, axes = plt.subplots(nrows=1, ncols=2)

        axes[0].matshow(M)
        axes[0].set_title('before')
        ax2 = axes[1].matshow(p_M)
        axes[1].set_title('after')
        
        plt.savefig(name + '.png')
        plt.cla()
        plt.close()

    def test_on_level(tree):
        if not os.path.exists(tmp_dic_path):
            tmp_dic = {}
        else:
            tmp_dic = np.load(tmp_dic_path, allow_pickle=True).item()

        def id(n): return [i.id for i in n.leaves]
        level_dic = {}
        cur_list = copy.deepcopy(tree.children)
        level_dic[1] = cur_list
        cur_level = 2
        while True:
            nxt_list = []
            for c in cur_list:
                if len(c.children)==0:
                    nxt_list.append(c)
                else:
                    nxt_list.extend(c.children)

            level_dic[cur_level] = nxt_list
            cur_level += 1
            cur_list = nxt_list

            if len(nxt_list)==len(cur_list):
                done = True
                for c in nxt_list:
                    if len(c.children)!=0:
                        done=False
                        break
                if done:
                    break
            

        max_level = max(level_dic.keys())
        
        level_CMs = []
        for l in range(1, max_level+1):
            print(l)
            nodes = level_dic[l]
            n = len(nodes)
            level_CM = [[0 for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    ci = nodes[i]
                    cj = nodes[j]
                    if (ci.name, cj.name) in tmp_dic:
                        num = tmp_dic[ci.name, cj.name]
                    elif (cj.name, ci.name) in tmp_dic:
                        num = tmp_dic[cj.name, ci.name]
                    else:
                        ci_ids = id(ci)
                        cj_ids = id(cj)
                        idx_1 = np.array([_ for _ in itertools.product(ci_ids, cj_ids)])
                        num_1 = np.sum(CM[idx_1[:, 0], idx_1[:, 1]])
                        idx_2 = np.array([_ for _ in itertools.product(cj_ids, ci_ids)])
                        num_2 = np.sum(CM[idx_2[:, 0], idx_2[:, 1]])                        
                        num = num_1 + num_2
                        tmp_dic[ci.name, cj.name] = num
                        tmp_dic[cj.name, ci.name] = num
                    level_CM[i][j] = num
                    level_CM[j][i] = level_CM[i][j]

            level_CM = np.array(level_CM)
            level_CMs.append(level_CM)
            np.save(tmp_dic_path, tmp_dic)

        return level_CMs

    def compare(level_CMs_before, level_CMs_after):

        def get_show_mat(m):
            show_m = m
            for i in range(len(m)):
                for j in range(len(m)):
                    if show_m[i][j]!=0:
                        show_m[i][j] = np.log(show_m[i][j])
            return show_m
        def draw_mat(m1, m2, name):
            fig, axes = plt.subplots(nrows=1, ncols=2)

            axes[0].matshow(m1)
            axes[0].set_title('before')
            ax2 = axes[1].matshow(m2)
            axes[1].set_title('after')
            
            plt.savefig(name + '.png')
            plt.cla()
            plt.close()

        num_level = len(level_CMs_before)
        max_conf = max([np.max(level_CMs_before[i]) for i in range(num_level)])
        max_conf += 1000

        before_bars = []
        after_bars = []
        for i in range(num_level):
            before_bars.append(Measure(max_conf - level_CMs_before[i]).get_measure()['BAR'])
            after_bars.append(Measure(max_conf - level_CMs_after[i]).get_measure()['BAR'])

            draw_mat(get_show_mat(level_CMs_before[i]), get_show_mat(level_CMs_after[i]), 'level_'+str(i))
        print('before')
        print(before_bars)
        print('after')
        print(after_bars)

    def compute_level_bar(level_CMs):
        num_level = len(level_CMs)
        max_conf = max([np.max(level_CMs[i]) for i in range(num_level)])
        max_conf += 1000
        bars = []
        for i in range(num_level):
            tmp_m = max_conf - level_CMs[i]
            for i in range(len(tmp_m)): tmp_m[i][i] = 0
            bars.append(Measure(tmp_m).get_measure()['BAR'])
        return bars
    
    def draw_log_cm(mat, img_path):
        show_mat = np.array(mat)
        n = len(show_mat)
        for i in range(n):
            for j in range(n):
                if show_mat[i][j]!=0:
                    show_mat[i][j] = np.log(show_mat[i][j])
        draw(show_mat, None, img_path)


    cnt = 0
    def reorder_tree_by_level(tree):
        nonlocal  cnt
        if len(tree.children)==0: return
        n = len(tree.children)
        if n>=2: 
            cnt += 1

            level_DM = [[0 for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    ci = tree.children[i]
                    cj = tree.children[j]
                    level_DM[i][j] = confusion_dic[ci.name, cj.name]
            
            # read b from file
            # bar_mat = getBARMatrix(len(level_DM), len(level_DM)-1)
            # write_case(n, bar_mat, level_DM, tmp_matrix_txt_path)
            # r_idx = getQAPReordering_file(0, False, tmp_matrix_txt_path, hierarchy_path, name2id_txt_path, restart, num_iters, False, avgs, stds)

            r_idx = getQAPReordering(0, False, len(level_DM), level_DM, hierarchy_path, name2id_txt_path, restart, num_iters, False, avgs, stds)

            r_children = []
            for i in range(n):
                r_children.append(tree.children[r_idx[i]])
            tree.children = r_children

            # print(tree.name, len(r_idx), '----------------')

            # if len(r_idx)>=10:
            #     show_m = np.array(level_DM)
            #     for i in range(len(level_DM)):
            #         for j in range(len(level_DM)):
            #             if show_m[i][j]!=0:
            #                 show_m[i][j] = np.log(show_m[i][j])
                
            #     draw_pair(show_m, r_idx, tree.name)


        for c in tree.children:
            reorder_tree_by_level(c)


    level_CMs_before = test_on_level(tree)

    # ori_ids = preOrder(tree)

    if use_level:
        start = time.time()
        reorder_tree_by_level(tree)
        end = time.time()
        print('对子树排序用时 ', end-start)
        setLeaves(tree)
        ord_ids = [i.id for i in tree.leaves]
    else:
        start = time.time()
        ord_ids = getQAPReordering_file(gamma, True, matrix_txt_path, hierarchy_path, name2id_txt_path, restart, num_iters, preorder, avgs, stds)
        end = time.time()
        print('对hierarchy排序用时 ', end-start)
    reorderTree(tree, ord_ids)
    setLeaves(tree)

    level_CMs_after = test_on_level(tree)

    bar_before = compute_level_bar(level_CMs_before)
    bar_after = compute_level_bar(level_CMs_after)
    if not preorder:
        for l, cm in enumerate(level_CMs_before):
            draw_log_cm(cm, osp.join(path, 'results', 'before_level_'+str(l)))
    for l, cm in enumerate(level_CMs_after):
        draw_log_cm(cm, osp.join(path, 'results', f'after_level_{l}_{suffix}'))    
    print(bar_before)
    print(bar_after)

    norm_bar_before = [(bar_before[i]-avgs[i+1])/stds[i+1] for i in range(len(bar_before))]
    norm_bar_after = [(bar_after[i]-avgs[i+1])/stds[i+1] for i in range(len(bar_after))]
    print('normed')
    print(norm_bar_before)
    print(norm_bar_after)
    print(np.average(norm_bar_before))
    print(np.average(norm_bar_after))

    if not os.path.exists(results_dic_path):
        dic = {}
    else:
        dic = np.load(results_dic_path, allow_pickle=True).item()
    
    dic['avgs'] = avgs[1:]
    if preorder:
        dic['before_pre'] = bar_before
    else:
        dic['before'] = bar_before
    dic[f'after_{suffix}'] = bar_after
    np.save(results_dic_path, dic)

    if use_level:
        write_pre_hierarchy(tree, hierarchy_pre_json_path, hierarchy_pre_txt_path)
    write_result(results_dic_path, results_csv_path, avgs[1:], stds[1:])
    return ord_ids

def getOrderedHierarchyQAPSA(confusion, use_level=False):
    path = '/data/fengyuan/72/ConfusionMatrix/backend/data/reordering/data/tmp/80'
    from .data.utils import get_max_depth, expand_tree, tree_to_json, hierarchy_json_to_txt, name2id_npy_to_txt, write_case
    from .pkgC.SA.agg_sa import getQAPReordering_file, getQAPReordering, getAvgAndStd

    name2id_npy_path = os.path.join(path, 'name2id.npy')
    name2id_txt_path = os.path.join(path, 'name2id.txt')
    hierarchy_json_path = os.path.join(path, 'hierarchy.json')
    hierarchy_txt_path = os.path.join(path, 'hierarchy.txt')
    hierarchy_pre_json_path = osp.join(path, 'hierarchy_pre.json')
    hierarchy_pre_txt_path = osp.join(path, 'hierarchy_pre.txt')
    matrix_txt_path = osp.join(path, 'matrix.txt')
    matrix_npy_path = osp.join(path, 'conf_mat.npy')
    avg_and_std_path = osp.join(path, 'avg_std.npy')

    CM = confusion['matrix'] 
    if type(CM)==list: CM  = np.array(CM)
    conf_mat = CM
    n = len(conf_mat)
    bar_mat = getBARMatrix(n, n-1)
    if not osp.exists(matrix_npy_path) or not osp.exists(matrix_txt_path):
        np.save(matrix_npy_path, conf_mat)
        write_case(n, bar_mat, conf_mat, matrix_txt_path)
    
    names = confusion['names']
    name2id = {name.replace(' ', '_'): id for id, name in enumerate(names)}
    if not osp.exists(name2id_npy_path) or not osp.exists(name2id_txt_path):
        np.save(name2id_npy_path, name2id)
        name2id_npy_to_txt(name2id_npy_path, name2id_txt_path)

    hierarchy = confusion['hierarchy']
    # for h in hierarchy:
    #     if len(h['children'])==1 and h['children'][0]==h['name']:
    #         h['children'] = []
    original_tree = hierarchyToTree(hierarchy)
    original_name2id = {name: id for id, name in enumerate(names)}
    prepareTree(original_tree, original_name2id)

    tree = original_json_to_tree({'name': 'root', 'children': hierarchy})
    prepareTree(tree, name2id)
    max_depth = [0]
    get_max_depth(tree, max_depth)
    expand_tree(tree, max_depth[0])
    setLeaves(tree)
    setDepth(tree, 0)

    js = tree_to_json(tree)
    if not osp.exists(hierarchy_json_path) or not osp.exists(hierarchy_txt_path):
        json.dump(js, open(hierarchy_json_path,'w',encoding="utf-8"))
        hierarchy_json_to_txt(hierarchy_json_path, hierarchy_txt_path)


    if not osp.exists(avg_and_std_path):
        avg_std = getAvgAndStd(True, matrix_txt_path, hierarchy_txt_path, name2id_txt_path, 5000)
        avgs = [0] + avg_std[0][::-1]
        stds = [0] + avg_std[1][::-1]
        avg_std = [avgs, stds]
        np.save(avg_and_std_path, avg_std)
    avg_std = np.load(avg_and_std_path, allow_pickle=True)
    avgs = avg_std[0]
    stds = avg_std[1]

    tmp_dic_path = osp.join(path, "tmp_dic.npy") 
    tmp_matrix_txt_path = osp.join(path, 'tmp_matrix.txt')

    confusion_dic = {}
    getConfusionDic(tree, CM, confusion_dic)

    # num_perm = calcPermutations(tree)
    # print('num_perm', num_perm)


    def test_on_level(tree):
        if not os.path.exists(tmp_dic_path):
            tmp_dic = {}
        else:
            tmp_dic = np.load(tmp_dic_path, allow_pickle=True).item()

        def id(n): return [i.id for i in n.leaves]
        level_dic = {}
        cur_list = copy.deepcopy(tree.children)
        level_dic[1] = cur_list
        cur_level = 2
        while True:
            nxt_list = []
            for c in cur_list:
                if len(c.children)==0:
                    nxt_list.append(c)
                else:
                    nxt_list.extend(c.children)

            level_dic[cur_level] = nxt_list
            cur_level += 1
            cur_list = nxt_list

            if len(nxt_list)==len(cur_list):
                done = True
                for c in nxt_list:
                    if len(c.children)!=0:
                        done=False
                        break
                if done:
                    break
            

        max_level = max(level_dic.keys())
        
        level_CMs = []
        for l in range(1, max_level+1):
            # print(l)
            nodes = level_dic[l]
            n = len(nodes)
            level_CM = [[0 for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    ci = nodes[i]
                    cj = nodes[j]
                    if (ci.name, cj.name) in tmp_dic:
                        num = tmp_dic[ci.name, cj.name]
                    elif (cj.name, ci.name) in tmp_dic:
                        num = tmp_dic[cj.name, ci.name]
                    else:
                        ci_ids = id(ci)
                        cj_ids = id(cj)
                        idx_1 = np.array([_ for _ in itertools.product(ci_ids, cj_ids)])
                        num_1 = np.sum(CM[idx_1[:, 0], idx_1[:, 1]])
                        idx_2 = np.array([_ for _ in itertools.product(cj_ids, ci_ids)])
                        num_2 = np.sum(CM[idx_2[:, 0], idx_2[:, 1]])                        
                        num = num_1 + num_2
                        tmp_dic[ci.name, cj.name] = num
                        tmp_dic[cj.name, ci.name] = num
                    level_CM[i][j] = num
                    level_CM[j][i] = level_CM[i][j]

            level_CM = np.array(level_CM)
            level_CMs.append(level_CM)
            np.save(tmp_dic_path, tmp_dic)

        return level_CMs

    def compute_level_bar(level_CMs):
        num_level = len(level_CMs)
        max_conf = max([np.max(level_CMs[i]) for i in range(num_level)])
        max_conf += 1000
        bars = []
        for i in range(num_level):
            tmp_m = max_conf - level_CMs[i]
            for i in range(len(tmp_m)): tmp_m[i][i] = 0
            bars.append(Measure(tmp_m).get_measure()['BAR'])
        return bars
    
    cnt = 0
    def reorder_tree_by_level(tree):
        nonlocal  cnt
        if len(tree.children)==0: return
        n = len(tree.children)
        if n>=2: 
            cnt += 1

            level_DM = [[0 for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    ci = tree.children[i]
                    cj = tree.children[j]
                    level_DM[i][j] = confusion_dic[ci.name, cj.name]
            
            # read b from file
            # bar_mat = getBARMatrix(len(level_DM), len(level_DM)-1)
            # write_case(n, bar_mat, level_DM, tmp_matrix_txt_path)
            # r_idx = getQAPReordering_file(0, False, tmp_matrix_txt_path, hierarchy_path, name2id_txt_path, restart, num_iters, False, avgs, stds)

            r_idx = getQAPReordering(0, False, len(level_DM), level_DM, hierarchy_txt_path, name2id_txt_path, 5, 10000, False, avgs, stds)

            r_children = []
            for i in range(n):
                r_children.append(tree.children[r_idx[i]])
            tree.children = r_children

        for c in tree.children:
            reorder_tree_by_level(c)


    # level_CMs_before = test_on_level(tree)


    if use_level:
        start = time.time()
        reorder_tree_by_level(tree)
        end = time.time()
        print('对子树排序用时 ', end-start)
        setLeaves(tree)
        ord_ids = [i.id for i in tree.leaves]
    else:
        start = time.time()
        ord_ids = getQAPReordering(1, True, len(CM), CM, hierarchy_txt_path, name2id_txt_path, 5, 10000, False, avgs, stds)
        # ord_ids = getQAPReordering(1, True, matrix_txt_path, hierarchy_txt_path, name2id_txt_path, 5, 10000, False, avgs, stds)
        end = time.time()
        print('对hierarchy排序用时 ', end-start)
    reorderTree(tree, ord_ids)
    setLeaves(tree)

    # level_CMs_after = test_on_level(tree)

    # bar_before = compute_level_bar(level_CMs_before)
    # bar_after = compute_level_bar(level_CMs_after) 
    # norm_bar_before = [(bar_before[i]-avgs[i+1])/stds[i+1] for i in range(len(bar_before))]
    # norm_bar_after = [(bar_after[i]-avgs[i+1])/stds[i+1] for i in range(len(bar_after))]

    # print(bar_before)
    # print(bar_after)
    # print('normed')
    # print(norm_bar_before)
    # print(norm_bar_after)
    # print(np.average(norm_bar_before))
    # print(np.average(norm_bar_after))

    if use_level:
        write_pre_hierarchy(tree, hierarchy_pre_json_path, hierarchy_pre_txt_path)


    reorderTree(original_tree, ord_ids)
    opt_hierarchy = treeToHierarchy(original_tree)

    # TODO: 层数大于2
    for i in range(len(opt_hierarchy)):
        if type(opt_hierarchy[i]) == str:
            opt_hierarchy[i] = { 'name': opt_hierarchy[i], 'children': [opt_hierarchy[i]] }

    return opt_hierarchy, ord_ids

def getOrderedHierarchyLevelKOLO(confusion, method="levelKOLO"):
    print('Using method ', method)

    CM = confusion['matrix']
    names = confusion['names']
    hierarchy = confusion['hierarchy']
    if type(CM)==list: CM  = np.array(CM)

    for h in hierarchy:
        if len(h['children'])==1 and h['children'][0]==h['name']:
            h['children'] = []

    name2id = {name: id for id, name in enumerate(names)}

    tree = hierarchyToTree(hierarchy)
    prepareTree(tree, name2id)

    DM = getDistanceMatrix(CM)
    DM = np.array(DM)

    distance_dic = {}
    getDistanceDic(tree, CM, distance_dic)

    n_leaves = len(DM)
    bind_width = len(CM) - 1
    mea_mat = getBARMatrix(n_leaves, bw=bind_width)
    ord_top_bar = 0
    ord_bottom_bar = 0

    # from .pkgC.ckolo import reordering
    # from .pkgR import seriation
    from .pkgC.genetic_reordering import genetic_reordering


    def reorder_tree_by_level(tree):
        nonlocal ord_top_bar
        if len(tree.children)==0: return
        n = len(tree.children)
        # TODO
        level_DM = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                ci = tree.children[i]
                cj = tree.children[j]
                level_DM[i][j] = distance_dic[ci.name, cj.name]
                level_DM[j][i] = level_DM[i][j]
        
        child_set = {'root': [c.name for c in tree.children]}
        # kolo
        # r_idx = reordering.kolo_reordering(level_DM, {}, child_set)
        # ga
        # result = genetic_reordering.geneticReordering(level_DM, {}, child_set, 1000, 50, 40, 4, -1, 800) 
        # sa
        # bar_mat = getBARMatrix(len(level_DM), len(level_DM)-1)
        # r_idx = qap_sa.getQAPReordering(level_DM, bar_mat, name2id, hierarchy)

        r_children = []
        for i in range(n):
            r_children.append(tree.children[r_idx[i]])
        tree.children = r_children

        ## correctness debug
        # r_m = np.array(level_DM)
        # r_m = r_m[r_idx][:,r_idx]
        # r_c = seriation.getMetric(r_m)
        # print('r_c', r_c['Path_length'])
        # rr_idx = seriation.getSeriationOrder(level_DM, "TSP")
        # rr_m = np.array(level_DM)
        # rr_m = rr_m[rr_idx][:,rr_idx]
        # rr_c = seriation.getMetric(rr_m)
        # print('rr_c', rr_c['Path_length'])
        # print(r_c['Path_length']<=rr_c['Path_length'])
        ##

        ## compare metrics of top level
        if tree.name=='root':
            print('ori top')
            r_mea = getMetric(level_DM, None)
            print(r_mea)
            print(Measure(level_DM).get_measure())

            print('ord top')
            top_dm = np.array(level_DM)[r_idx][:, r_idx]
            print(getMetric(top_dm, None))
            print(Measure(top_dm).get_measure())
            ord_top_bar = Measure(top_dm).get_measure()
        ##


        for c in tree.children:
            reorder_tree_by_level(c)

    ori_ids = tree.preOrder()
    # print('before reorder: ', ori_ids)

    reorder_tree_by_level(tree)
    ord_ids = tree.preOrder()
    # print('after reorder: ', ord_ids)

    opt_hierarchy = treeToHierarchy(tree)

    for i in range(len(opt_hierarchy)):
        if type(opt_hierarchy[i]) == str:
            opt_hierarchy[i] = { 'name': opt_hierarchy[i], 'children': [opt_hierarchy[i]] }

    
    ## compare metrics of bottom level
    print('ori bottom')
    print(getMetric(DM, None))
    print(Measure(DM).get_measure())

    print('ord bottom')
    print(getMetric(DM, ord_ids))
    bottom_dm = np.array(DM)[ord_ids][:, ord_ids]
    print(Measure(bottom_dm).get_measure())
    ord_bottom_bar = Measure(bottom_dm).get_measure()
    return opt_hierarchy, ord_ids, ord_top_bar, ord_bottom_bar

def getOrderedHierarchyCKOLO(confusion, agg=1):
    """
        agg: considering path
    """
    CM = confusion['matrix'] 
    if type(CM)==list: CM  = np.array(CM)
    names = confusion['names']
    hierarchy = confusion['hierarchy']

    for h in hierarchy:
        if len(h['children'])==1 and h['children'][0]==h['name']:
            h['children'] = []

    child_set = handleHierarchy(hierarchy)
    name2id = {name: id for id, name in enumerate(names)}

    DM = getDistanceMatrix(CM)
    DM = np.array(DM)
    n_leaves = len(CM)

    bind_width = len(CM) - 1
    mea_mat = getBARMatrix(n_leaves, bw=bind_width)

    print('ori bar', calcBAR(DM, mea_mat))

    tree = hierarchyToTree(hierarchy)
    prepareTree(tree, name2id)

    aggDM = getAggDM(CM, DM, tree)

    import time
    from .pkgC.ckolo import reordering
    start = time.time()

    if agg==1:
        ord_ids = reordering.kolo_reordering(aggDM, name2id, child_set)
    elif agg==0:
        ord_ids = reordering.kolo_reordering(DM, name2id, child_set)
    end = time.time()

    print('kolo reordering time: ', end-start)

    tree = hierarchyToTree(hierarchy)
    setIds(tree, name2id)
    setLeaves(tree)
    reorderTree(tree, ord_ids)
    setLeaves(tree)
    opt_hierarchy = treeToHierarchy(tree)

    for i in range(len(opt_hierarchy)):
        if type(opt_hierarchy[i]) == str:
            opt_hierarchy[i] = { 'name': opt_hierarchy[i], 'children': [opt_hierarchy[i]] }


    print('ori bottom')
    print(Measure(DM).get_measure(bw=-1))

    print('ord bottom')
    print(Measure(DM[ord_ids][:, ord_ids]).get_measure(bw=-1))

    print('ord top')
    top_dm = getTopDM(CM, tree)
    print(Measure(top_dm).get_measure(bw=-1))
    # top_mea_mat = getBARMatrix(len(top_dm), bw=2)
    # print('ord top bar', calcBAR(top_dm, top_mea_mat))
    
    print('kolo ord_ids', ord_ids)
    # print('kolo VAR', calcBAR(DM, mea_mat, ord_ids))
    return opt_hierarchy, ord_ids

def getOrderedHierarchyR(confusion, method="OLO"):
    print("Using method " + method)

    CM = confusion['matrix']
    names = confusion['names']
    hierarchy = confusion['hierarchy']
    if type(CM)==list: CM  = np.array(CM)

    for h in hierarchy:
        if len(h['children'])==1 and h['children'][0]==h['name']:
            h['children'] = []

    name2id = {name: id for id, name in enumerate(names)}

    tree = hierarchyToTree(hierarchy)
    prepareTree(tree, name2id)

    DM = getDistanceMatrix(CM)
    DM = np.array(DM)

    n_leaves = len(DM)
    bind_width = len(CM) - 1
    mea_mat = getBARMatrix(n_leaves, bw=bind_width)
    ord_top_bar = 0
    ord_bottom_bar = 0

    distance_dic = {}
    getDistanceDic(tree, CM, distance_dic)

    # from .pkgR import seriation
    from .pkgC.qap_test import QapOrder
    
    def reorder_tree_by_level(tree):
        nonlocal ord_top_bar
        if len(tree.children)==0: return
        n = len(tree.children)
        level_DM = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                ci = tree.children[i]
                cj = tree.children[j]
                level_DM[i][j] = distance_dic[ci.name, cj.name]
                level_DM[j][i] = level_DM[i][j]
        # r
        # r_idx = seriation.getSeriationOrder(level_DM, method)
        # r_idx = seriation.getBiclustOrder(level_DM, method)
        # qap heu
        bar_mat = getBARMatrix(len(level_DM), len(level_DM)-1)
        # r_idx = QapOrder.getFantOrder(level_DM, bar_mat)
        # r_idx = QapOrder.getQapOrder(level_DM, bar_mat)
        r_idx = QapOrder.getQAPReordering(len(level_DM), bar_mat, level_DM, {}, {})
        # r_idx = QapOrder.getTabuOrder(level_DM, bar_mat)
        # r_idx = qap_sa.getSAOrder(level_DM, bar_mat)
        # tmp_idx = [0 for i in range(len(r_idx))]
        # for idx, i in enumerate(r_idx): tmp_idx[i] = idx
        # for i in range(len(r_idx)): r_idx[i] = tmp_idx[i]

        r_children = []
        for i in range(n):
            r_children.append(tree.children[r_idx[i]])
        tree.children = r_children

        ## compare metrics of top level
        if tree.name=='root':
            print('ori top')
            r_mea = getMetric(level_DM, None)
            print(r_mea)
            print(Measure(level_DM).get_measure())

            print('ord top')
            top_dm = np.array(level_DM)[r_idx][:, r_idx]
            print(getMetric(top_dm, None))
            print(Measure(top_dm).get_measure())
            ord_top_bar = Measure(top_dm).get_measure()
        ##

        for c in tree.children:
            reorder_tree_by_level(c)

    ori_ids = tree.preOrder()
    # print('before reorder: ', ori_ids)

    reorder_tree_by_level(tree)
    ord_ids = tree.preOrder()
    # print('after reorder: ', ord_ids)

    opt_hierarchy = treeToHierarchy(tree)

    for i in range(len(opt_hierarchy)):
        if type(opt_hierarchy[i]) == str:
            opt_hierarchy[i] = { 'name': opt_hierarchy[i], 'children': [opt_hierarchy[i]] }

    ## compare metrics of bottom level
    print('ori bottom')
    print(getMetric(DM, None))
    print(Measure(DM).get_measure())

    print('ord bottom')
    print(getMetric(DM, ord_ids))
    bottom_dm = np.array(DM)[ord_ids][:, ord_ids]
    print(Measure(bottom_dm).get_measure())
    ord_bottom_bar = Measure(bottom_dm).get_measure()
    print(ord_ids)
    return opt_hierarchy, ord_ids, ord_top_bar, ord_bottom_bar

def getOrderedHierarchyGENE(confusion, agg=0):
    agg = 1
    CM = confusion['matrix']
    CM  = np.array(CM)
    names = confusion['names']
    hierarchy = confusion['hierarchy']

    name2class = {}
    for i in range(len(hierarchy)):
        c = hierarchy[i]
        for name in c['children']:
            name2class[name] = i

    hierarchy = confusion['hierarchy']
    for h in hierarchy:
        if len(h['children'])==1 and h['children'][0]==h['name']:
            h['children'] = []
    child_set = handleHierarchy(hierarchy)
    name2id = {name: id for id, name in enumerate(names)}

    id2name = dict([(v,k) for (k,v) in name2id.items()])
    id2class = dict([(k, name2class[v]) for (k,v) in id2name.items()])
    # print('id2class', id2class)

    DM = getDistanceMatrix(CM)
    DM = np.array(DM)
    n_leaves = len(CM)

    dim = len(DM)
    bw = 16
    mea_mat = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if abs(i-j)<=bw:
                mea_mat[i][j] = bw+1-abs(i-j)
            else:
                mea_mat[i][j] = 0
    for i in range(dim): mea_mat[i][i] = 0

    ol = 0
    for i in range(dim):
        for j in range(dim):
            ol += DM[i][j]*mea_mat[i][j]
    print('original bar', ol)

    tree = hierarchyToTree(hierarchy)

    prepareTree(tree, name2id)

    # aggCM = [[0 for _ in range(n_leaves)] for _ in range(n_leaves)]
    # dic_aggcm_inter = {}
    # for n1 in tree.leaves:
    #     for n2 in tree.leaves:
    #         if n1==n2: continue
    #         aggCM[n1.id][n2.id] = getAggCM(n1, n2, CM, dic_aggcm_inter)
    # aggCM = np.array(aggCM)

    aggDM = [[0 for _ in range(n_leaves)] for _ in range(n_leaves)]
    dic_aggdm_inter = {}
    for n1 in tree.leaves:
        for n2 in tree.leaves:
            if n1==n2: continue
            aggDM[n1.id][n2.id] = calcAggDM(n1, n2, CM, DM, dic_aggdm_inter)
    aggDM = np.array(aggDM)

    import time
    from .pkgC.genetic_reordering import genetic_reordering
    start = time.time()

    if agg==1:
        result = genetic_reordering.geneticReordering(aggDM, name2id, child_set, 1000, 50, 40, 4, -1, 800, mea_mat) # 800 2273
        ord_ids = result[0]
        record = result[1]
    elif agg==0:
        # int _epoch, int _population, int _num_hybrid, int _mutate_prob, int _num_adj, int _random_seed
        # result = genetic_reordering.geneticReordering(DM, name2id, child_set, 50000, 50, 40, 6, -1, 800) 312168
        ## 312163 50000, 100, 60, 4, -1, 800 mutate all
        #  312053 mutate_new 0.25 result = genetic_reordering.geneticReordering(DM, name2id, child_set, 180000, 50, 40, 4, -1, 800) 
        result = genetic_reordering.geneticReordering(DM, name2id, child_set, 1000, 50, 40, 4, -1, 800, mea_mat) 
        ord_ids = result[0]
        record = result[1]
    end = time.time()

    import matplotlib.pyplot as plt
    import datetime
    plt.plot(record)
    plt.savefig(f"gene-{datetime.datetime.now().strftime('%R-%S')}.png")

    print('GA c++ time: ', end-start)
    print('ord_ids', ord_ids)

    tree = hierarchyToTree(hierarchy)
    setIds(tree, name2id)
    setLeaves(tree)

    reorderTree(tree, ord_ids)

    setLeaves(tree)

    opt_hierarchy = treeToHierarchy(tree)
    for i in range(len(opt_hierarchy)):
        if type(opt_hierarchy[i]) == str:
            opt_hierarchy[i] = { 'name': opt_hierarchy[i], 'children': [opt_hierarchy[i]] }

    
    # compute metrics
    print('ori bottom')
    r_mea = getMetric(DM, None, r_method='ori')
    print(r_mea)
    mea = Measure(DM).get_measure()
    print(mea)

    print('ord bottom')
    r_mea = getMetric(DM, ord_ids, r_method="KOLO")
    print(r_mea)
    bottom_dm = DM[ord_ids][:, ord_ids]
    mea = Measure(bottom_dm).get_measure()
    print(mea)

    print('ord top')
    def get_level_dm(node):
        n = len(node.children)
        level_DM = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                ci = node.children[i]
                cj = node.children[j]
                level_DM[i][j] = distance_dic[ci.name, cj.name]
                level_DM[j][i] = level_DM[i][j]
        return level_DM
    distance_dic = {}
    getDistanceDic(tree, CM, distance_dic)
    top_dm = get_level_dm(tree)
    r_mea = getMetric(top_dm, None, r_method='KOLO')
    print(r_mea)
    # print('top_dm')
    # for line in top_dm: print(line)
    mea = Measure(top_dm).get_measure()
    print(mea)
    top_dim = len(top_dm)
    bw = 3
    top_mea_mat = np.zeros((top_dim, top_dim))
    for i in range(top_dim):
        for j in range(top_dim):
            if abs(i-j)<=bw:
                top_mea_mat[i][j] = bw+1-abs(i-j)
            else:
                top_mea_mat[i][j] = 0
    for i in range(top_dim): top_mea_mat[i][i] = 0
    ol = 0
    for i in range(top_dim):
        for j in range(top_dim):
            ol += top_dm[i][j]*top_mea_mat[i][j]
    print('ord top bar', ol)
    
    # n=12
    # conf = np.zeros((n,n))
    # for i in range(n):
    #     for j in range(n):
    #         conf[i][j] = 1000 - top_dm[i][j]
    # for line in conf: print(line)

    ol = 0
    for i in range(dim):
        for j in range(dim):
            ol += DM[ord_ids[i]][ord_ids[j]]*mea_mat[i][j]
    print('original bar', ol)
    
    return opt_hierarchy, ord_ids

def getBARMatrix(dim=80, bw=16):
    mea_mat = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if abs(i-j)<=bw:
                mea_mat[i][j] = bw+1-abs(i-j)
            else:
                mea_mat[i][j] = 0
    for i in range(dim): mea_mat[i][i] = 0
    return mea_mat


"""
    Measuring
"""
def getMetric(DM, perm):
    """
        DM:       distance matrix (2d list/numpy)
        perm:     order (list)
        r_method: reordering method (str)
        metric:   (str/None)
    """
    if type(DM)==list: DM = np.array(DM)
    if perm: DM = DM[perm][:, perm]

    # from .pkgR import seriation
    results = seriation.getMetric(DM.tolist())
    return results


"""
    Computing distance/similarity
"""
def calcBAR(DM, BAR, perm=None):
    if type(DM)==list: DM = np.array(DM)
    if perm: DM = DM[perm][:, perm]
    dim = len(DM)
    bar = 0
    for i in range(dim):
        for j in range(dim):
            bar += DM[i][j] * BAR[i][j]
    return bar

def getTopDM(CM, tree):
    def get_level_dm(node):
        n = len(node.children)
        level_DM = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                ci = node.children[i]
                cj = node.children[j]
                level_DM[i][j] = distance_dic[ci.name, cj.name]
                level_DM[j][i] = level_DM[i][j]
        return level_DM
    distance_dic = {}
    getDistanceDic(tree, CM, distance_dic)
    top_dm = get_level_dm(tree)
    return top_dm

def getAggDM(CM, DM, tree):
    n_leaves = len(CM)
    aggDM = [[0 for _ in range(n_leaves)] for _ in range(n_leaves)]
    dic_aggdm_inter = {}
    for n1 in tree.leaves:
        for n2 in tree.leaves:
            if n1==n2: continue
            aggDM[n1.id][n2.id] = calcAggDM(n1, n2, CM, DM, dic_aggdm_inter)
    aggDM = np.array(aggDM)
    return aggDM

def getConfusionDic(tree, CM, dic):
    """
        计算两个中间节点的混淆数目
        同一个节点与自己的混淆数目为0
    """
    if len(tree.children)==0: return
    def id(n): return [i.id for i in n.leaves]
    import itertools
    for i in range(len(tree.children)):
        for j in range(len(tree.children)):
            if i==j:
                dic[tree.children[i].name, tree.children[j].name] = 0
            else:
                ci = tree.children[i]
                cj = tree.children[j]
                ci_ids = id(ci)
                cj_ids = id(cj)
                        
                idx_1 = np.array([_ for _ in itertools.product(ci_ids, cj_ids)])
                num_1 = np.sum(CM[idx_1[:, 0], idx_1[:, 1]])

                dic[ci.name, cj.name] = num_1
    for c in tree.children:
        getConfusionDic(c, CM, dic)

def getDistanceDic(tree, CM, dic):
    """
        compute distance between children of an interval nodes
    """
    if len(tree.children)==0: return
    def id(n): return [i.id for i in n.leaves]
    import itertools
    for i in range(len(tree.children)):
        for j in range(i+1, len(tree.children)):
            ci = tree.children[i]
            cj = tree.children[j]
            ci_ids = id(ci)
            cj_ids = id(cj)
                    
            idx_1 = np.array([_ for _ in itertools.product(ci_ids, cj_ids)])
            num_1 = np.sum(CM[idx_1[:, 0], idx_1[:, 1]])
            idx_2 = np.array([_ for _ in itertools.product(cj_ids, ci_ids)])
            num_2 = np.sum(CM[idx_2[:, 0], idx_2[:, 1]])
            
            num = num_1 + num_2
            if(num>MAX_DIS):
                raise NotImplementedError
                print()
            # dic[ci.name, cj.name] = dic[cj.name, ci.name] = 1 / (1+num)
            dic[ci.name, cj.name] = dic[cj.name, ci.name] = MAX_DIS-num
    for c in tree.children:
        getDistanceDic(c, CM, dic)

def getDistanceMatrix(CM):
    """return distance matrix of confusion matrix CM
    """
    n = len(CM)
    DM = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            # dis = 1 / 1 + (CM[i][j] + CM[j][i])
            dis = MAX_DIS - (CM[i][j] + CM[j][i])
            DM[i][j] = dis
            DM[j][i] = DM[i][j]
    return DM

def calcAggDM(n1, n2, CM, DM, mem):
    # DM: distance matrix of leaves
    # CM: confusion matrix of leaves
    # n1, n2: either leaves or interval nodes

    def getDis(n1, n2):
        if len(n1.children)==0 and len(n2.children)==0: return -CM[n1.id, n2.id]
        if (n1.name, n2.name) in mem: return mem[n1.name, n2.name]
        if (n2.name, n1.name) in mem: return mem[n2.name, n1.name]
        def id(n): return [i.id for i in n.leaves]

        n1_leaves_id = id(n1)
        n2_leaves_id = id(n2)

        import itertools
        idx_1 = np.array([_ for _ in itertools.product(n1_leaves_id, n2_leaves_id)])
        dis_1 = - np.sum(CM[idx_1[:, 0], idx_1[:, 1]])
        idx_2 = np.array([_ for _ in itertools.product(n2_leaves_id, n1_leaves_id)])
        dis_2 = - np.sum(CM[idx_2[:, 0], idx_2[:, 1]])
        dis = dis_1 + dis_2

        mem[n1.name, n2.name] = mem[n2.name, n1.name] = dis
        return dis

    c = 1
    gamma = 2
    dis = 0
    if n1.depth!=n2.depth:
        dis = getDis(n1, n2)
        c *= gamma
        while n1.depth > n2.depth:
            n1 = n1.parent
        while n2.depth > n1.depth:
            n2 = n2.parent 
    while n1!=n2:
        dis += c * getDis(n1, n2)
        c *= gamma
        n1 = n1.parent
        n2 = n2.parent
    return dis

def getAggCM(n1, n2, CM, mem):
    # DM: distance matrix of leaves
    # CM: confusion matrix of leaves
    # n1, n2: either leaves or interval nodes

    def getDis(n1, n2):
        if n1.isLeaf() and n2.isLeaf(): return CM[n1.id, n2.id]
        if (n1.name, n2.name) in mem: return mem[n1.name, n2.name]
        def id(n): return [i.id for i in n.leaves]

        n1_leaves_id = id(n1)
        n2_leaves_id = id(n2)

        import itertools
        idx_1 = np.array([_ for _ in itertools.product(n1_leaves_id, n2_leaves_id)])
        dis_1 = np.sum(CM[idx_1[:, 0], idx_1[:, 1]])

        mem[n1.name, n2.name] = dis_1
        return dis_1

    c = 1
    gamma = 2
    dis = 0
    if n1.depth!=n2.depth:
        dis = getDis(n1, n2)
        c *= gamma
        while n1.depth > n2.depth:
            n1 = n1.parent
        while n2.depth > n1.depth:
            n2 = n2.parent 
    while n1!=n2:
        dis += c * getDis(n1, n2)
        c *= gamma
        n1 = n1.parent
        n2 = n2.parent
    return dis

"""
    Handling hierarchy
"""

def handleHierarchy(hierarchy):
    """data structure for ckolo
    """
    child_set = {}
    def travel(node):
        if len(node['children'])==0:
            return
        if isinstance(node['children'][0], str):
            child_set[node['name']] = node['children']
        else:
            child_set[node['name']] = [c['name'] for c in node['children']]
            for c in node['children']: travel(c)
    travel({'name': 'root', 'children': hierarchy})
    return child_set

def treeToHierarchy(tree):
    hierarchy = []
    def dfs(node):
        if len(node.children)==0: return node.name
        item = {}
        item['name'] = node.name
        item['children'] = [dfs(c) for c in node.children]
        return item
    hierarchy = [dfs(c) for c in tree.children]
    return hierarchy

def hierarchyToTree(hierarchy):
    def buildTree(item):
        if isinstance(item, str): return TreeNode(name=item)
        node = TreeNode(name=item['name'])
        for child in item['children']:
            if type(child)==str and child==node.name:
                continue
            node.children.append(buildTree(child))
        return node
    tree = buildTree({'name': 'root', 'children': hierarchy})
    return tree

def original_json_to_tree(js):
    if type(js)==str:
        name = js.replace(' ', '_')
        node = TreeNode(name=name)
    else:
        ori_name = js['name']
        name = ori_name.replace(' ', '_')
        node = TreeNode(name=name)
        if 'children' in js:
            for c in js['children']:
                if type(c)==str and c.replace(' ', '_')==node.name:
                    continue
                c_tree = original_json_to_tree(c)
                c_tree.parent = node
                node.children.append(c_tree)
    return node

def setIds(tree, name2id):
    """Set id of leaves according to 'name2id'.
    """
    global INTERVAL_NODE_START_ID
    start_id = INTERVAL_NODE_START_ID
    que = deque([tree])
    while len(que) > 0:
        p = que.popleft()
        if len(p.children)==0:
            p.id = name2id[p.name]
        else:
            p.id = start_id
            start_id += 1
        for c in p.children:
            que.append(c)

def setParent(node):
    if node.isLeaf():
        return
    for c in node.children:
        c.parent = node
        setParent(c)

def setLeaves(node):
    if len(node.children)==0:
        node.leaves = [node]
        return
    node.leaves = []
    for c in node.children:
        setLeaves(c)
        node.leaves.extend(c.leaves)

def setDepth(node, depth):
    node.depth = depth
    for c in node.children:
        setDepth(c, depth+1)

def prepareTree(tree, name2id):
    setIds(tree, name2id)
    setLeaves(tree)
    setParent(tree)
    tree.parent = tree
    setDepth(tree, 0)

def reorderTree(node, ord_ids):
    """Reorder tree according to perm of leaves.
    """
    if len(node.children)==0:
        return
    node.children.sort(key=lambda c: ord_ids.index(c.leaves[0].id))
    for c in node.children: reorderTree(c, ord_ids)


def calcPermutations(tree):
    
    childrens = []
    def travel(node):
        childrens.append(node)
        for c in node.children:
            travel(c)
    travel(tree)
    cnt = 1
    for c in childrens:
        n = len(c.children)
        if n<=1: continue
        if n*(n-1)==0:
            print()
        cnt += (n*(n-1))/2
    return cnt

def write_pre_hierarchy(tree, hierarchy_json_path, hierarchy_txt_path):
    # from .read_hierarchy import tree_to_json, tree2chierarchy
    from .data.imagenet.read_hierarchy import tree_to_json, hierarchy_json_to_txt

    js = tree_to_json(tree)
    json.dump(js, open(hierarchy_json_path,'w',encoding="utf-8"))
    hierarchy_json_to_txt(hierarchy_json_path, hierarchy_txt_path)

def write_result(results_dic_path, results_csv_path, avgs, stds):
    dic = np.load(results_dic_path, allow_pickle=True).item()
    data = [v for v in dic.values()]
    indexs = [k for k in dic.keys()]
    norm_data = copy.deepcopy(data)
    for i in range(len(data)):
        for j in range(len(data[0])):
            norm_data[i][j] = (norm_data[i][j] - avgs[j]) / stds[j]
    data += norm_data
    indexs += indexs
    df = pd.DataFrame(data, index=indexs,dtype=float)
    df.to_csv(results_csv_path)

    print()
 

"""
    test correctness of KOLO 
"""

def test_kolo_1d():
    from .pkgC.ckolo import reordering
    
    n = 15

    # seriation test
    for cnt in range(100):
        dm = np.random.randint(500, size=(n,n))
        for i in range(n): dm[i][i] = 0
        dm = dm + dm.T
        child_set = {'root': [str(i) for i in range(n)]}

        # kolo
        k_idx = reordering.kolo_reordering(dm, {}, child_set)
        k_mat = dm.copy()
        k_mat = k_mat[k_idx][:,k_idx]
        k_mec = seriation.getMetric(k_mat)
        print('k_mec', k_mec['Path_length'])

        # pkg r
        r_idx = seriation.getSeriationOrder(dm, "OLO")
        r_mat = dm.copy()
        r_mat = r_mat[r_idx][:,r_idx]
        r_mec = seriation.getMetric(r_mat)
        print('r_mec', r_mec['Path_length'])

        # enumerate
        # import itertools
        # perms = itertools.permutations([i for i in range(len(dm))])
        # p_idx = None
        # min_dis = 1e30
        # for perm in perms:
        #     # print(perm)
        #     tmp_dis = 0
        #     for i in range(len(perm)-1):
        #         tmp_dis += dm[perm[i]][perm[i+1]]
        #     if tmp_dis < min_dis:
        #         min_dis = tmp_dis
        #         p_idx = perm
        # p_idx = list(p_idx)
        # p_mat = dm.copy()
        # p_mat = p_mat[p_idx][:,p_idx]
        # p_mec = seriation.getMetric(p_mat)
        # print('p_mec', p_mec['Path_length'])
        
        # assert(k_mec['Path_length']==p_mec['Path_length'])      
        assert(k_mec['Path_length']<=r_mec['Path_length'])

    print()

def test_kolo_hierarchy():
    from .pkgC.ckolo import reordering
    for cnt in range(100):
        k1 = 3
        k2 = 4
        def init_tree():
            root = TreeNode(name='root')
            i1 = TreeNode(name='i1')
            i2 = TreeNode(name='i2')
            i3 = TreeNode(name='i3')
            # i4 = TreeNode(name='i4')
            root.children = [i1, i2, i3]
            c = []
            for i in range(k1*k2):
                c.append(TreeNode(name=str(i)))
            i1.children = c[:4]
            i2.children = c[4:8]
            i3.children = c[8:12]
            # i4.children = c[9:]

            name2id = {}
            for i in range(k1*k2): name2id[str(i)] = i
            prepareTree(root, name2id)
            return root, name2id
        root, name2id = init_tree()
        dm = np.random.randint(500, size=(k1*k2,k1*k2))
        for i in range(k1*k2): dm[i][i] = 0
        dm = dm + dm.T
        hierarchy = treeToHierarchy(root)
        child_set= handleHierarchy(hierarchy)
        k_idx = reordering.kolo_reordering(dm, name2id, child_set)
        k_mat = dm.copy()
        k_mat = k_mat[k_idx][:,k_idx]
        k_mec = seriation.getMetric(k_mat)
        print('k_mec', k_mec['Path_length'])

        import itertools
        indexs = list(itertools.permutations([i for i in range(k1)]))
        p_idx = None
        min_dis = 1e30
        perms = []
        seg_perms = []
        for i in range(k1):
            seg_perms.append(list(itertools.permutations([i for i in range(i*k2, (i+1)*k2)])))
        seg_perms = np.array(seg_perms)
        
        for idx in indexs:
            tmp_seg_perms = seg_perms[list(idx)]
            k2_e = len(tmp_seg_perms[0])
            for i in range(k2_e**k1):
                tmp = []
                for j in range(k1):
                    tmp.extend(tmp_seg_perms[j][int((i/(k2_e**j))%k2_e)])
                perms.append(tmp)


        for perm in perms:
            tmp_dis = 0
            for i in range(len(perm)-1):
                tmp_dis += dm[perm[i]][perm[i+1]]
            if tmp_dis < min_dis:
                min_dis = tmp_dis
                p_idx = perm
        p_idx = list(p_idx)
        p_mat = dm.copy()
        p_mat = p_mat[p_idx][:,p_idx]
        p_mec = seriation.getMetric(p_mat)
        print('p_mec', p_mec['Path_length'])

        assert(p_mec['Path_length']==k_mec['Path_length'])

    print()    

