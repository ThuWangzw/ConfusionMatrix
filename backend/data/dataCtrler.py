import os
import json
import copy
import bisect
from tkinter import N
from matplotlib.pyplot import box
import numpy as np
import pickle
import torch
import math
import io
from PIL import Image
import logging
import base64
from queue import PriorityQueue

from data.grid.sampling import HierarchySampling
from data.grid.gridLayout import GridLayout
from data.fisher import get_split_pos
from data.drawBox import Annotator

class DataCtrler(object):

    def __init__(self):
        super().__init__()
        self.iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
        self.classID2Idx = {}      
        self.hierarchy = {}  
        self.names = []
        self.grider = GridLayout()

    def process(self, rawDataPath, bufferPath, reordered=True):
        """process raw data
        - rawDataPath/
          - images/
          - labels/
          - predicts/
          - meta.json
        """        
        # init paths
        self.root_path = rawDataPath        
        self.images_path = os.path.join(self.root_path, "images")
        self.labels_path = os.path.join(self.root_path, "labels")
        self.predicts_path = os.path.join(self.root_path, "predicts")
        self.meta_path = os.path.join(self.root_path, "meta.json")
        self.features_path = os.path.join(self.root_path, "pr_features")
        self.gt_features_path = os.path.join(self.root_path, "gt_features")
        if not os.path.exists(self.features_path):
            os.makedirs(self.features_path)
        if not os.path.exists(self.gt_features_path):
            os.makedirs(self.gt_features_path)
        if not os.path.exists(bufferPath):
            os.makedirs(bufferPath)
        setting_name = os.path.basename(os.path.normpath(rawDataPath))
        self.raw_data_path = os.path.join(bufferPath, "{}_raw_data.pkl".format(setting_name))
        self.label_predict_iou_path = os.path.join(bufferPath, "{}_predict_label_iou.pkl".format(setting_name))
        self.box_size_split_path = os.path.join(bufferPath, "{}_box_size_split.pkl".format(setting_name))
        self.box_size_dist_path = os.path.join(bufferPath, "{}_box_size_dist.pkl".format(setting_name))
        self.box_aspect_ratio_split_path = os.path.join(bufferPath, "{}_box_aspect_ratio_split.pkl".format(setting_name))
        self.box_aspect_ratio_dist_path = os.path.join(bufferPath, "{}_box_aspect_ratio_dist.pkl".format(setting_name))
        self.hierarchy_sample_path = os.path.join(bufferPath, "{}_hierarchy_samples.pkl".format(setting_name))
        self.all_features_path = os.path.join(bufferPath, "{}_features.npy".format(setting_name))
        self.eval_data = torch.load(os.path.join(rawDataPath, 'eval.pth'))
        
        
        self.logger = logging.getLogger('dataCtrler')

        # read raw data
        if os.path.exists(self.raw_data_path):
            with open(self.raw_data_path, 'rb') as f:
                self.image2index, self.raw_labels, self.raw_label2imageid, self.imageid2raw_label, self.raw_predicts, self.raw_predict2imageid, self.imageid2raw_predict = pickle.load(f)
        else:
            self.image2index = {}
            id=0
            for name in os.listdir(self.images_path):
                self.image2index[name.split('.')[0]]=id
                id += 1
            ## read raw labels
            # format: label, box(cx, cy, w, h), isCrowd(0/1)
            self.raw_labels = np.zeros((0,6), dtype=np.float32)
            self.raw_label2imageid = np.zeros(0, dtype=np.int32)
            self.imageid2raw_label = np.zeros((id, 2), dtype=np.int32)
            for imageName in os.listdir(self.labels_path):
                label_path = os.path.join(self.labels_path, imageName)
                imageid = self.image2index[imageName.split('.')[0]]
                with open(label_path) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any([len(x)>6 for x in lb]):
                        #TODO segmentation
                        pass
                    lb = np.array(lb, dtype=np.float32)
                    self.imageid2raw_label[imageid][0] = len(self.raw_labels)
                    self.imageid2raw_label[imageid][1] = len(self.raw_labels)+len(lb)
                    if len(lb)>0:
                        self.raw_labels = np.concatenate((self.raw_labels, lb), axis=0)
                        self.raw_label2imageid = np.concatenate((self.raw_label2imageid, np.ones(len(lb), dtype=np.int32)*imageid))
                    
                    
            ## read raw predicts
            # format: predict, confidence, box(cx, cy, w, h)
            self.raw_predicts = np.zeros((0,6), dtype=np.float32)
            self.raw_predict2imageid = np.zeros(0, dtype=np.int32)
            self.imageid2raw_predict = np.zeros((id, 2), dtype=np.int32)
            for imageName in os.listdir(self.predicts_path):
                predict_path = os.path.join(self.predicts_path, imageName)
                imageid = self.image2index[imageName.split('.')[0]]
                with open(predict_path) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any([len(x)>6 for x in lb]):
                        #TODO segmentation
                        pass
                    lb = np.array(lb, dtype=np.float32)
                    self.imageid2raw_predict[imageid][0] = len(self.raw_predicts)
                    self.imageid2raw_predict[imageid][1] = len(self.raw_predicts)+len(lb)
                    if len(lb)>0:
                        self.raw_predicts = np.concatenate((self.raw_predicts, lb), axis=0)
                        self.raw_predict2imageid = np.concatenate((self.raw_predict2imageid, np.ones(len(lb), dtype=np.int32)*imageid))
            with open(self.raw_data_path, 'wb') as f:
                pickle.dump((self.image2index, self.raw_labels, self.raw_label2imageid, self.imageid2raw_label, self.raw_predicts, self.raw_predict2imageid, self.imageid2raw_predict), f)
        self.index2image = ['']*len(self.image2index)
        for image, index in self.image2index.items():
            self.index2image[index] = image
        
        ## init meta data
        with open(self.meta_path) as f:
            metas = json.load(f)
            categorys = metas["categories"]
            for classIdx in range(len(categorys)):
                self.classID2Idx[categorys[classIdx]["id"]] = classIdx
                self.names.append(categorys[classIdx]["name"])
                superCategory = categorys[classIdx]["supercategory"]
                if superCategory not in self.hierarchy:
                    self.hierarchy[superCategory] = {
                        "name": superCategory,
                        "children": []
                    }
                self.hierarchy[superCategory]["children"].append(categorys[classIdx]["name"])
            self.hierarchy = list(self.hierarchy.values())
            self.hierarchy.append({
                    "name": "background",
                    "children": ["background"]
                })
            self.names.append("background")
            self.classID2Idx[-1]=len(categorys)
                
            
        # compute (prediction, label) pair
        # creates a map, with different IoU threshold (0.5~0.95 0.05) as key and (predict_label_pairs, iou) as value
        # do not store the unmatched gt here, because different confidence thershold may result in different "Missed Error"
        if os.path.exists(self.label_predict_iou_path):
            with open(self.label_predict_iou_path, 'rb') as f:
                self.pairs_map_under_iou_thresholds = pickle.load(f)
        else:
            self.pairs_map_under_iou_thresholds = self.compute_label_predict_pair()
            with open(self.label_predict_iou_path, 'wb') as f:
                pickle.dump(self.pairs_map_under_iou_thresholds, f)
        
        if os.path.exists(self.box_size_split_path):
            with open(self.box_size_split_path, 'rb') as f:
                self.box_size_split_map = pickle.load(f)
        else:
            self.box_size_split_map = {}
        
        if os.path.exists(self.box_aspect_ratio_split_path):
            with open(self.box_aspect_ratio_split_path, 'rb') as f:
                self.box_aspect_ratio_split_map = pickle.load(f)
        else:
            self.box_aspect_ratio_split_map = {}
        
        # init size
        self.label_size = self.raw_labels[:,3]*self.raw_labels[:,4]
        self.predict_size = self.raw_predicts[:,4]*self.raw_predicts[:,5]

        # get box size distribution in [0,1]
        if os.path.exists(self.box_size_dist_path):
            with open(self.box_size_dist_path, 'rb') as f:
                self.box_size_dist_map = pickle.load(f)
        else:
            self.box_size_dist_map = None
            self.box_size_dist_map = self.getBoxSizeDistribution()
            with open(self.box_size_dist_path, 'wb') as f:
                pickle.dump(self.box_size_dist_map, f)

        # init aspect ratio
        self.label_aspect_ratio = self.raw_labels[:,3]/self.raw_labels[:,4]
        self.predict_aspect_ratio = self.raw_predicts[:,4]/self.raw_predicts[:,5]

        # get box aspect ratio distribution
        if os.path.exists(self.box_aspect_ratio_dist_path):
            with open(self.box_aspect_ratio_dist_path, 'rb') as f:
                self.box_aspect_ratio_dist_map = pickle.load(f)
        else:
            self.box_aspect_ratio_dist_map = None
            self.box_aspect_ratio_dist_map = self.getBoxAspectRatioDistribution()
            with open(self.box_aspect_ratio_dist_path, 'wb') as f:
                pickle.dump(self.box_aspect_ratio_dist_map, f)

        # direction map, also use IoU threshold as key because different match results in different directions
        self.directions_map = {}
        for iou_thres in self.iou_thresholds:
            predict_label_pairs, _ = self.pairs_map_under_iou_thresholds[iou_thres]
            directionIdxes = np.where(np.logical_and(predict_label_pairs[:,0]>-1, predict_label_pairs[:,1]>-1))[0]
            directionVectors = self.raw_predicts[predict_label_pairs[directionIdxes,0]][:,[2,3]] - self.raw_labels[predict_label_pairs[directionIdxes,1]][:,[1,2]]
            directionNorm = np.sqrt(np.power(directionVectors[:,0], 2)+ np.power(directionVectors[:,1], 2))
            directionCos = directionVectors[:,0]/directionNorm
            directions = np.zeros(directionCos.shape[0], dtype=np.int32)
            directionSplits = np.array([math.cos(angle/180*math.pi) for angle in [180, 157.5, 112.5, 67.5, 22.5, 0]])
            for i in range(2,len(directionSplits)):
                directions[np.logical_and(directionCos>directionSplits[i-1], directionCos<=directionSplits[i])] = i-1
            negaYs = np.logical_and(directionVectors[:,1]<0, directions!=0)
            directions[negaYs] = 8-directions[negaYs]
            # use box w, h to define min_shift, with a maximum value of 0.05
            min_shift = (self.raw_labels[predict_label_pairs[directionIdxes, 1], 3] + \
                self.raw_labels[predict_label_pairs[directionIdxes, 1], 4]).squeeze() / 10
            min_shift[min_shift > 0.05] = 0.05
            directions[directionNorm<min_shift] = 8
            self.directions_map[iou_thres] = -1*np.ones(predict_label_pairs.shape[0], dtype=np.int32)
            self.directions_map[iou_thres][directionIdxes] = directions
        
        # read feature data
        if os.path.exists(self.all_features_path):
            all_features = np.load(self.all_features_path)
            self.pr_features = all_features[:len(self.raw_predicts)]
            self.gt_features = all_features[len(self.raw_predicts):]
        else:
            self.pr_features = np.zeros((self.raw_predicts.shape[0], 256))
            for name in os.listdir(self.images_path):
                feature_path = os.path.join(self.features_path, name.split('.')[0]+'.npy')
                imageid = self.image2index[name.split('.')[0]]
                boxCount = self.imageid2raw_predict[imageid][1]-self.imageid2raw_predict[imageid][0]
                if not os.path.exists(feature_path):
                    # WARNING
                    self.logger.warning("can't find feature: %s" % feature_path)
                    self.pr_features[self.imageid2raw_predict[imageid][0]:self.imageid2raw_predict[imageid][1]] = np.random.rand(boxCount, 256)
                else:
                    self.pr_features[self.imageid2raw_predict[imageid][0]:self.imageid2raw_predict[imageid][1]] = np.load(feature_path)
            
            self.gt_features = np.zeros((self.raw_labels.shape[0], 256))
            for name in os.listdir(self.images_path):
                feature_path = os.path.join(self.gt_features_path, name.split('.')[0]+'.npy')
                imageid = self.image2index[name.split('.')[0]]
                boxCount = self.imageid2raw_label[imageid][1]-self.imageid2raw_label[imageid][0]
                if not os.path.exists(feature_path):
                    # WARNING
                    self.logger.warning("can't find feature: %s" % feature_path)
                    self.gt_features[self.imageid2raw_label[imageid][0]:self.imageid2raw_label[imageid][1]] = np.random.rand(boxCount, 256)
                else:
                    self.gt_features[self.imageid2raw_label[imageid][0]:self.imageid2raw_label[imageid][1]] = np.load(feature_path)
            np.save(self.all_features_path, np.concatenate((self.pr_features, self.gt_features)))
        
        # hierarchy sampling
        self.sampler = HierarchySampling()
        if os.path.exists(self.hierarchy_sample_path):
            self.sampler.load(self.hierarchy_sample_path)
        else:
            # fit all features (predict & gt) to sampler
            labels = np.concatenate((self.raw_predicts[:, 0], self.raw_labels[:, 0])).astype(np.int32)
            features = np.concatenate((self.pr_features, self.gt_features))
            self.sampler.fit(features, labels, 400)
            self.sampler.dump(self.hierarchy_sample_path)
          
    def getMetaData(self):
        return {
            "hierarchy": self.hierarchy,
            "names": self.names
        }
            
    def compute_label_predict_pair(self):

        def compute_per_image(detections, labels, pos_thres, bg_thres=0.1, max_det=100):
            pr_bbox = detections[:, 2:6]
            pr_cat = detections[:, 0].astype(np.int32)
            pr_conf = detections[:, 1]
            gt_bbox = labels[:, 1:5]
            gt_iscrowd = labels[:,5].astype(np.int32)
            gt_cat = labels[:,0].astype(np.int32)
            pr_uni_cat = np.unique(detections[:, 0].astype(np.int32))
            pr_type = np.zeros(len(detections), dtype=np.int32) # record different type of detections, e.g., TP, dup, confusion, etc.
            # -1 for ignored, 0 for abandoned, 1 for TP, 2~6 for Cls(confusion), Loc, Cls+Loc, Dup, Bkgd
            gt_match = -np.ones(len(labels), dtype=np.int32)
            pr_match = -np.ones(len(detections), dtype=np.int32)
            iou_pair = cal_iou(pr_bbox, gt_bbox, gt_iscrowd)
            pr_abandon = np.zeros(0, dtype=np.int32) # abandon the predictions of each category out of maxdet
            # within category match => TP
            for _cat in pr_uni_cat:
                # get detections with category _cat
                pr_idx = np.where(pr_cat == _cat)[0]
                gt_idx = np.where(gt_cat == _cat)[0]
                if len(gt_idx) == 0:
                    continue
                # sort by detection confidence, select a maximum of max_det
                pr_idx = pr_idx[np.argsort(-pr_conf[pr_idx])]
                if len(pr_idx) > max_det:
                    pr_abandon = np.concatenate((pr_abandon, pr_idx[max_det:]))
                    pr_idx = pr_idx[:max_det]
                # sort by isCrowd attr, put crowd to the back
                gt_idx = gt_idx[np.argsort(gt_iscrowd[gt_idx])]
                for _pr_idx in pr_idx:
                    t = pos_thres
                    m = -1
                    # the same as pycocotool
                    for _gt_idx in gt_idx:
                        if gt_match[_gt_idx] >= 0 and gt_iscrowd[_gt_idx] == 0:
                            continue
                        if m > -1 and gt_iscrowd[m] == 0 and gt_iscrowd[_gt_idx] == 1:
                            break
                        iou = iou_pair[_pr_idx, _gt_idx]
                        if iou < t:
                            continue
                        t = iou
                        m = _gt_idx
                    if m == -1:
                        continue
                    gt_match[m] = _pr_idx
                    pr_match[_pr_idx] = m
                    if gt_iscrowd[m]:
                        pr_type[_pr_idx] = -1 # ignore predictions matched with crowd
                    else:
                        pr_type[_pr_idx] = 1 # TP
            # inter category match for FP
            # get all unmatched predictions that were not abandoned
            pr_idx = np.setdiff1d(np.where(pr_match == -1)[0], pr_abandon)
            pr_idx = pr_idx[np.argsort(-pr_conf[pr_idx])]
            # exclude all iscrowd gt, this accords with TIDE
            gt_idx = np.where(gt_iscrowd == 0)[0]
            for _pr_idx in pr_idx:
                if len(gt_idx) == 0:
                    pr_type[_pr_idx] = 6 # background error, for no gt in the image
                    continue
                t = -1
                m = -1
                # find gt with largest IoU
                for _gt_idx in gt_idx:
                    iou = iou_pair[_pr_idx, _gt_idx]
                    if iou < t:
                        continue
                    t = iou
                    m = _gt_idx
                if t < bg_thres: # background
                    pr_type[_pr_idx] = 6
                elif t > pos_thres: # right location
                    pr_match[_pr_idx] = m
                    if gt_cat[m] == pr_cat[_pr_idx]: # duplicate prediction
                        pr_type[_pr_idx]  = 5
                    else: # confusion
                        pr_type[_pr_idx]  = 2
                        if gt_match[m] == -1:
                            gt_match[m] = _pr_idx # mark gt as used
                else: # Loc error
                    if gt_cat[m] == pr_cat[_pr_idx]: # Location error
                        pr_type[_pr_idx]  = 3
                        pr_match[_pr_idx] = m
                        if gt_match[m] == -1:
                            gt_match[m] = _pr_idx
                    else: # Class + Location error
                        # TODO: now consider this kind as background error
                        # pr_match[_pr_idx] = m
                        pr_type[_pr_idx]  = 4
                        # do not mark gt as used here
            # should ignore detections matched with ignored gt
            ret_ious = np.zeros(0)
            ret_match = -1*np.ones((0, 2), dtype=np.int32)
            for _pr_idx in range(len(detections)):
                if pr_type[_pr_idx] <= 0:
                    continue
                ret_match = np.concatenate((ret_match, np.array([[_pr_idx, pr_match[_pr_idx]]])))
                if pr_match[_pr_idx] == -1:
                    ret_ious = np.concatenate((ret_ious, [0]))
                else:
                    ret_ious = np.concatenate((ret_ious, [iou_pair[_pr_idx, pr_match[_pr_idx]]]))
            return ret_match, ret_ious
        self.pairs_map_under_iou_thresholds = {}
        bg_thres = 0.1
        for pos_thres in self.iou_thresholds:
            # remember: this doesn't contain all predictions, because some are ignored !!!
            predict_label_pairs = -1*np.ones((0, 2), dtype=np.int32)
            predict_label_ious = np.zeros(0)
            for imageidx in range(len(self.image2index)):
                matches, ious = compute_per_image(self.raw_predicts[self.imageid2raw_predict[imageidx][0]:self.imageid2raw_predict[imageidx][1]],
                                            self.raw_labels[self.imageid2raw_label[imageidx][0]:self.imageid2raw_label[imageidx][1]], pos_thres, bg_thres)
                negaWeights = np.where(matches[:,1]==-1)[0]
                if len(matches)>0:
                    matches[:,1]+=self.imageid2raw_label[imageidx][0]
                    matches[:,0]+=self.imageid2raw_predict[imageidx][0]
                    matches[negaWeights,1]=-1
                    predict_label_pairs = np.concatenate((predict_label_pairs, matches))
                    predict_label_ious = np.concatenate((predict_label_ious, ious))
            self.pairs_map_under_iou_thresholds[pos_thres] = (predict_label_pairs, predict_label_ious)
        return self.pairs_map_under_iou_thresholds
    
    def filterSamples(self, query = None):
        """
            return filtered, unmatch_predict, unmatch_label: index of pairs in predict_label_pairs
        """
        default_query = {
            "label_size": [0,1],
            "predict_size": [0,1],
            "label_aspect_ratio": [0,110],
            "predict_aspect_ratio": [0,110],
            "direction": [0,1,2,3,4,5,6,7,8],
            "label": np.arange(len(self.classID2Idx)-1),
            "predict": np.arange(len(self.classID2Idx)-1),
            "split": 10,
            "conf_thres": 0.05,
            "iou_thres": 0.5
        }
        iou_thres = self.iou_thresholds[0]
        conf_thres = 0.05
        if query is not None:
            query = {**default_query, **query}
            iou_thres = query['iou_thres']
            conf_thres = query['conf_thres']
        
        # separate matched pairs from unmatch ones
        # index of pred_label_pairs
        predict_label_pairs, _ = self.pairs_map_under_iou_thresholds[iou_thres]
        filtered = np.arange(len(predict_label_pairs))[np.logical_and(predict_label_pairs[:,1]>-1, self.raw_predicts[predict_label_pairs[:,0], 1]>conf_thres)]
        unmatch_predict = np.arange(len(predict_label_pairs))[np.logical_and(predict_label_pairs[:,1]==-1, self.raw_predicts[predict_label_pairs[:,0], 1]>conf_thres)]
        # this is index of gt, not pred_label_pairs, as this is calculated after filtering
        # should ignore iscrowd object
        unmatch_label = np.setdiff1d(np.arange(len(self.raw_labels))[self.raw_labels[:, 5]==0], predict_label_pairs[filtered, 1])

        if query is not None:
            label_selected = np.array([True for _ in range(len(self.label_size))])
            predict_selected = np.array([True for _ in range(len(self.predict_size))])
            # size
            if not isinstance(query["label_size"][0], str) and not isinstance(query["label_size"][1], str):
                label_selected = np.logical_and(label_selected, np.logical_and(self.label_size>=query["label_size"][0], self.label_size<=query["label_size"][1]))
            if not isinstance(query["predict_size"][0], str) and not isinstance(query["predict_size"][1], str):
                predict_selected = np.logical_and(predict_selected, np.logical_and(self.predict_size>=query["predict_size"][0], self.predict_size<=query["predict_size"][1]))
            # aspect ratio
            if not isinstance(query["label_aspect_ratio"][0], str) and not isinstance(query["label_aspect_ratio"][1], str):
                label_selected = np.logical_and(label_selected, np.logical_and(self.label_aspect_ratio>=query["label_aspect_ratio"][0], self.label_aspect_ratio<=query["label_aspect_ratio"][1]))
            if not isinstance(query["predict_aspect_ratio"][0], str) and not isinstance(query["predict_aspect_ratio"][1], str):
                predict_selected = np.logical_and(predict_selected, np.logical_and(self.predict_aspect_ratio>=query["predict_aspect_ratio"][0], self.predict_aspect_ratio<=query["predict_aspect_ratio"][1]))
            # label
            label_selected = np.logical_and(label_selected, np.isin(self.raw_labels[:,0], query["label"]))
            predict_selected = np.logical_and(predict_selected, np.isin(self.raw_predicts[:,0], query["predict"]))

            # get results after query
            label_selected = np.arange(len(self.raw_labels))[label_selected]
            predict_selected = np.arange(len(self.raw_predicts))[predict_selected]
            filtered = filtered[np.isin(predict_label_pairs[filtered][:,1], label_selected)]
            filtered = filtered[np.isin(predict_label_pairs[filtered][:,0], predict_selected)]
            unmatch_predict = unmatch_predict[np.isin(predict_label_pairs[unmatch_predict][:,0], predict_selected)]
            # this is different because unmatch_label is NOT the index of predict_label_pairs
            unmatch_label = unmatch_label[np.isin(unmatch_label, label_selected)]

            # only for unmatch_label
            if isinstance(query["predict_size"][0], str) or isinstance(query["predict_size"][1], str) or \
               isinstance(query["predict_aspect_ratio"][0], str) or isinstance(query["predict_aspect_ratio"][1], str) or \
               -1 in query["predict"]:
               return np.array([], dtype=np.int32), np.array([], dtype=np.int32), unmatch_label
            # only for unmatch_predict
            if isinstance(query["label_size"][0], str) or isinstance(query["label_size"][1], str) or \
               isinstance(query["label_aspect_ratio"][0], str) or isinstance(query["label_aspect_ratio"][1], str) or \
               -1 in query["label"]:
               return np.array([], dtype=np.int32), unmatch_predict, np.array([], dtype=np.int32)
        
        return filtered, unmatch_predict, unmatch_label
    
    def getStatisticsMatrices(self, matrix, query):
        """
            matrix: a 3-d list consists of lists of indexes from predict_label_pairs 
        """
        statistics_modes = ['count']
        if query is not None and "return" in query:
            statistics_modes = query['return']
        iou_thres = self.iou_thresholds[0]
        if query is not None and "iou_thres" in query:
            iou_thres = query["iou_thres"]
        predict_label_pairs, predict_label_ious = self.pairs_map_under_iou_thresholds[iou_thres]
        bg_size_dist_bins = 5
        function_map = { # x represents array of indexes of predict_label_pair
            'count': lambda x: len(x),
            'avg_label_size': lambda x: 0 if predict_label_pairs[x[0],1]==-1 else self.label_size[predict_label_pairs[x, 1]].mean(),
            'avg_predict_size': lambda x: 0 if predict_label_pairs[x[0],0]==-1 else self.predict_size[predict_label_pairs[x, 0]].mean(),
            'avg_iou': lambda x: predict_label_ious[x].mean(),
            'avg_acc': lambda x: 0 if np.any(predict_label_pairs[x[0],:]==-1) else 
                            (self.raw_predicts[predict_label_pairs[x,0], 0]==self.raw_labels[predict_label_pairs[x,1], 0]).mean(),
            'avg_label_aspect_ratio': lambda x: 0 if predict_label_pairs[x[0],1]==-1 else self.label_aspect_ratio[predict_label_pairs[x, 1]].mean(),
            'avg_predict_aspect_ratio': lambda x: 0 if predict_label_pairs[x[0],0]==-1 else self.predict_aspect_ratio[predict_label_pairs[x, 0]].mean(),
            'direction': lambda x: [int(np.count_nonzero(self.directions_map[iou_thres][x]==i)) for i in range(9)],
            'size_comparison': lambda x: [0, 0] if len(x)==0 \
                else np.bincount((self.predict_size[predict_label_pairs[x, 0]]*bg_size_dist_bins).tolist()).tolist() if predict_label_pairs[x[0],1]==-1 \
                else np.bincount((self.label_size[predict_label_pairs[x, 1]]*bg_size_dist_bins).tolist()).tolist() if predict_label_pairs[x[0],0]==-1 else \
                [int(np.count_nonzero(self.predict_size[predict_label_pairs[x, 0]] > (self.label_size[predict_label_pairs[x, 1]]*1.15))),
                int(np.count_nonzero(self.label_size[predict_label_pairs[x, 1]] > (self.predict_size[predict_label_pairs[x, 0]]*1.15)))]
        }
        ret_matrixes = []
        for statistics_mode in statistics_modes:
            if statistics_mode not in function_map:
                raise NotImplementedError()
            map_func = function_map[statistics_mode]
            stat_matrix = np.zeros((len(matrix), len(matrix[0])), dtype=np.float64).tolist()
            for i in range(len(stat_matrix)):
                for j in range(len(stat_matrix[0])):
                    if statistics_mode != 'direction' and statistics_mode != 'size_comparison' and len(matrix[i][j]) == 0:
                        continue
                    # specially deal with unmatch gt
                    if j == len(stat_matrix[0]) - 1:
                        if statistics_mode == 'count':
                            stat_matrix[i][j] = len(matrix[i][j])
                        elif statistics_mode == 'avg_label_size':
                            stat_matrix[i][j] = self.label_size[matrix[i][j]].mean()
                        elif statistics_mode == 'avg_label_aspect_ratio':
                            stat_matrix[i][j] = self.label_aspect_ratio[matrix[i][j]].mean()
                        elif statistics_mode == 'direction':
                            stat_matrix[i][j] = [0 for _ in range(9)]
                        elif statistics_mode =='size_comparison':
                            stat_matrix[i][j] = np.bincount((self.label_size[matrix[i][j]]*bg_size_dist_bins).tolist()).tolist()
                            if len(stat_matrix[i][j]) > bg_size_dist_bins:
                                stat_matrix[i][j][bg_size_dist_bins-1] += stat_matrix[i][j][bg_size_dist_bins]
                                stat_matrix[i][j] = stat_matrix[i][j][:bg_size_dist_bins]
                        continue
                    if statistics_mode == 'direction' and (i == len(stat_matrix)-1 or j == len(stat_matrix[0])-1):
                        stat_matrix[i][j] = [0 for _ in range(9)]
                    else:
                        stat_matrix[i][j] = map_func(matrix[i][j])
                        if statistics_mode == 'size_comparison':
                            if len(stat_matrix[i][j]) > bg_size_dist_bins:
                                stat_matrix[i][j][bg_size_dist_bins-1] += stat_matrix[i][j][bg_size_dist_bins]
                                stat_matrix[i][j] = stat_matrix[i][j][:bg_size_dist_bins]
            ret_matrixes.append(np.array(stat_matrix).tolist()) # to avoid JSON unserializable error
        return ret_matrixes

    def getConfusionMatrix(self, query = None):
        """filtered confusion matrix

        Args:
            querys (dict): {label/predict size:[a, b], label/predict aspect_ratio:[a, b], direction: [0,..,8],
            label/predict: np.arange(80)}
        """
        filtered , unmatch_predict, unmatch_label = self.filterSamples(query)
        iou_thres = self.iou_thresholds[0]
        if query is not None and "iou_thres" in query:
            iou_thres = query["iou_thres"]
        predict_label_pairs, _ = self.pairs_map_under_iou_thresholds[iou_thres]
        label_target, pred_target = np.arange(80), np.arange(80)
        if query is not None and "label" in query and "predict" in query:
            label_target = query["label"]
            pred_target = query["predict"]
        confusion = [[[] for _ in range(len(pred_target)+1)] for _ in range(len(label_target)+1)]
        label_rec, pred_rec = [], []
        for i in range(len(label_target)):
            label_rec.append(self.raw_labels[predict_label_pairs[filtered][:, 1], 0]==label_target[i])
        for j in range(len(pred_target)):
            pred_rec.append(self.raw_predicts[predict_label_pairs[filtered][:, 0], 0]==pred_target[j])
        for i in range(len(label_target)):
            for j in range(len(pred_target)):
                confusion[i][j] = filtered[np.logical_and(label_rec[i], pred_rec[j])]
        for i in range(len(label_target)):
            confusion[i][len(pred_target)] = unmatch_label[self.raw_labels[unmatch_label, 0]==label_target[i]]
        for j in range(len(pred_target)):
            confusion[len(label_target)][j] = unmatch_predict[self.raw_predicts[predict_label_pairs[unmatch_predict][:, 0], 0]==pred_target[j]]
        return self.getStatisticsMatrices(confusion, query)
    
    def getOverallDistribution(self):
        ret_dict = {}
        K = 100
        data_dict = {
            'labelSize': self.label_size,
            'predictSize': self.predict_size,
            'labelAspectRatio': self.label_aspect_ratio,
            'predictAspectRatio': self.predict_aspect_ratio
        }
        for name, data in data_dict.items():
            tmp = sorted(data)
            ret_dict[name] = [float(tmp[i*(len(tmp)-1)//(K-1)]) for i in range(K)]
        return ret_dict
    
    def getZoomInDistribution(self, query):
        assert "query_key" in query
        target = query["query_key"]
        target_range = query["range"]
        query[target] = target_range
        K = 25
        split_pos = np.array([target_range[0]+i*(target_range[1]-target_range[0])/K for i in range(K+1)])
        filtered, unmatch_predict, unmatch_label = self.filterSamples(query)
        iou_thres = self.iou_thresholds[0]
        if query is not None and "iou_thres" in query:
            iou_thres = query["iou_thres"]
        predict_label_pairs, _ = self.pairs_map_under_iou_thresholds[iou_thres]
        label_target, pred_target = np.arange(len(self.classID2Idx)-1), np.arange(len(self.classID2Idx)-1)
        if target == 'label_size':
            all_dist = [0 for _ in range(K)]
            for i in range(K):
                last = split_pos[i]
                cur = split_pos[i+1]
                all_dist[i] = np.count_nonzero(np.logical_and(self.label_size>last, self.label_size<=cur))
            match_select = self.label_size[predict_label_pairs[filtered,1]]
            unmatch_select = self.label_size[unmatch_label]
            select_dist = [0 for _ in range(K)]
            for i in range(K):
                last = split_pos[i]
                cur = split_pos[i+1]
                select_dist[i] = np.count_nonzero(np.logical_and(match_select>last, match_select<=cur)) + \
                    np.count_nonzero(np.logical_and(unmatch_select>last, unmatch_select<=cur))
            confusion = [[[0 for _ in range(K)] for _ in range(len(pred_target)+1)] for _ in range(len(label_target)+1)]
            for (pr, gt) in predict_label_pairs[filtered]:
                confusion[int(self.raw_labels[gt, 0])][int(self.raw_predicts[pr, 0])][bisect.bisect_left(split_pos[1:-1], self.label_size[gt])]+=1
            for gt in unmatch_label:
                confusion[int(self.raw_labels[gt, 0])][len(pred_target)][bisect.bisect_left(split_pos[1:-1], self.label_size[gt])]+=1
        elif target == 'predict_size':
            all_dist = [0 for _ in range(K)]
            for i in range(K):
                last = split_pos[i]
                cur = split_pos[i+1]
                all_dist[i] = np.count_nonzero(np.logical_and(self.predict_size>last, self.predict_size<=cur))
            match_select = self.predict_size[predict_label_pairs[filtered,0]]
            unmatch_select = self.predict_size[predict_label_pairs[unmatch_predict,0]]
            select_dist = [0 for _ in range(K)]
            for i in range(K):
                last = split_pos[i]
                cur = split_pos[i+1]
                select_dist[i] = np.count_nonzero(np.logical_and(match_select>last, match_select<=cur)) + \
                    np.count_nonzero(np.logical_and(unmatch_select>last, unmatch_select<=cur))
            confusion = [[[0 for _ in range(K)] for _ in range(len(pred_target)+1)] for _ in range(len(label_target)+1)]
            for (pr, gt) in predict_label_pairs[filtered]:
                confusion[int(self.raw_labels[gt, 0])][int(self.raw_predicts[pr, 0])][bisect.bisect_left(split_pos[1:-1], self.predict_size[pr])]+=1
            for (pr, gt) in predict_label_pairs[unmatch_predict]:
                confusion[len(label_target)][int(self.raw_predicts[pr, 0])][bisect.bisect_left(split_pos[1:-1], self.predict_size[pr])]+=1
        elif target == 'label_aspect_ratio':
            all_dist = [0 for _ in range(K)]
            for i in range(K):
                last = split_pos[i]
                cur = split_pos[i+1]
                all_dist[i] = np.count_nonzero(np.logical_and(self.label_aspect_ratio>last, self.label_aspect_ratio<=cur))
            match_select = self.label_aspect_ratio[predict_label_pairs[filtered,1]]
            unmatch_select = self.label_aspect_ratio[unmatch_label]
            select_dist = [0 for _ in range(K)]
            for i in range(K):
                last = split_pos[i]
                cur = split_pos[i+1]
                select_dist[i] = np.count_nonzero(np.logical_and(match_select>last, match_select<=cur)) + \
                    np.count_nonzero(np.logical_and(unmatch_select>last, unmatch_select<=cur))
            confusion = [[[0 for _ in range(K)] for _ in range(len(pred_target)+1)] for _ in range(len(label_target)+1)]
            for (pr, gt) in predict_label_pairs[filtered]:
                confusion[int(self.raw_labels[gt, 0])][int(self.raw_predicts[pr, 0])][bisect.bisect_left(split_pos[1:-1], self.label_aspect_ratio[gt])]+=1
            for gt in unmatch_label:
                confusion[int(self.raw_labels[gt, 0])][len(pred_target)][bisect.bisect_left(split_pos[1:-1], self.label_aspect_ratio[gt])]+=1
        elif target == 'predict_aspect_ratio':
            all_dist = [0 for _ in range(K)]
            for i in range(K):
                last = split_pos[i]
                cur = split_pos[i+1]
                all_dist[i] = np.count_nonzero(np.logical_and(self.predict_aspect_ratio>last, self.predict_aspect_ratio<=cur))
            match_select = self.predict_aspect_ratio[predict_label_pairs[filtered,0]]
            unmatch_select = self.predict_aspect_ratio[predict_label_pairs[unmatch_predict,0]]
            select_dist = [0 for _ in range(K)]
            for i in range(K):
                last = split_pos[i]
                cur = split_pos[i+1]
                select_dist[i] = np.count_nonzero(np.logical_and(match_select>last, match_select<=cur)) + \
                    np.count_nonzero(np.logical_and(unmatch_select>last, unmatch_select<=cur))
            confusion = [[[0 for _ in range(K)] for _ in range(len(pred_target)+1)] for _ in range(len(label_target)+1)]
            for (pr, gt) in predict_label_pairs[filtered]:
                confusion[int(self.raw_labels[gt, 0])][int(self.raw_predicts[pr, 0])][bisect.bisect_left(split_pos[1:-1], self.predict_aspect_ratio[pr])]+=1
            for (pr, gt) in predict_label_pairs[unmatch_predict]:
                confusion[len(label_target)][int(self.raw_predicts[pr, 0])][bisect.bisect_left(split_pos[1:-1], self.predict_aspect_ratio[pr])]+=1
        else:
            raise NotImplementedError()

        return {
            'allDist': all_dist,
            'selectDist': select_dist,
            'confusion': confusion,
            'split': split_pos.tolist()
        }

   
    def getBoxSizeDistribution(self, query = None):
        """
            return label_size, pred_size distribution
        """
        if query is None and self.box_size_dist_map is not None:
            return self.box_size_dist_map
        filtered, unmatch_predict, unmatch_label = self.filterSamples(query)
        iou_thres = self.iou_thresholds[0]
        if query is not None and "iou_thres" in query:
            iou_thres = query["iou_thres"]
        predict_label_pairs, _ = self.pairs_map_under_iou_thresholds[iou_thres]
        filtered_label_size = self.label_size[predict_label_pairs[filtered][:,1]]
        filtered_predict_size = self.predict_size[predict_label_pairs[filtered][:,0]]
        unmatched_label_size = self.label_size[unmatch_label]
        unmatched_predict_size = self.predict_size[predict_label_pairs[unmatch_predict][:,0]]
        
        K = 25
        if query is not None and 'label_range' in query:
            label_range = query['label_range']
            predict_range = query['predict_range']
        else:
            maximum_val = np.max(np.concatenate((self.label_size, self.predict_size)))
            label_range = [0, maximum_val]
            predict_range = [0, maximum_val]
        label_split = np.array([label_range[0]+i*(label_range[1]-label_range[0])/K for i in range(K+1)])
        predict_split = np.array([predict_range[0]+i*(predict_range[1]-predict_range[0])/K for i in range(K+1)])
        
        label_box_size_dist, predict_box_size_dist = [0 for _ in range(K)], [0 for _ in range(K)]
        for i in range(K):
            label_box_size_dist[i] = np.count_nonzero(np.logical_and(filtered_label_size>label_split[i], filtered_label_size<=label_split[i+1])) + \
             np.count_nonzero(np.logical_and(unmatched_label_size>label_split[i], unmatched_label_size<=label_split[i+1]))
            predict_box_size_dist[i] = np.count_nonzero(np.logical_and(filtered_predict_size>predict_split[i], filtered_predict_size<=predict_split[i+1])) + \
             np.count_nonzero(np.logical_and(unmatched_predict_size>predict_split[i], unmatched_predict_size<=predict_split[i+1]))
            
        label_target, pred_target = np.arange(len(self.classID2Idx)-1), np.arange(len(self.classID2Idx)-1)
        label_box_size_confusion = [[[0 for _ in range(K)] for _ in range(len(pred_target)+1)] for _ in range(len(label_target)+1)]
        predict_box_size_confusion = [[[0 for _ in range(K)] for _ in range(len(pred_target)+1)] for _ in range(len(label_target)+1)]
        for (pr, gt) in predict_label_pairs[filtered]:
            if label_range[0] <= self.label_size[gt] <= label_range[1]:
                label_box_size_confusion[int(self.raw_labels[gt, 0])][int(self.raw_predicts[pr, 0])][bisect.bisect_left(label_split[1:-1], self.label_size[gt])]+=1
            if predict_range[0] <= self.predict_size[pr] <= predict_range[1]:
                predict_box_size_confusion[int(self.raw_labels[gt, 0])][int(self.raw_predicts[pr, 0])][bisect.bisect_left(predict_split[1:-1], self.predict_size[pr])]+=1
        for gt in unmatch_label:
            if label_range[0] <= self.label_size[gt] <= label_range[1]:
                label_box_size_confusion[int(self.raw_labels[gt, 0])][len(pred_target)][bisect.bisect_left(label_split[1:-1], self.label_size[gt])]+=1
        for (pr, gt) in predict_label_pairs[unmatch_predict]:
            if predict_range[0] <= self.predict_size[pr] <= predict_range[1]:
                predict_box_size_confusion[len(label_target)][int(self.raw_predicts[pr, 0])][bisect.bisect_left(predict_split[1:-1], self.predict_size[pr])]+=1
                
        return {
            'labelSizeAll': label_box_size_dist,
            'predictSizeAll': predict_box_size_dist,
            'labelSizeConfusion': label_box_size_confusion,
            'predictSizeConfusion': predict_box_size_confusion,
            'labelSplit': label_split.tolist(),
            'predictSplit': predict_split.tolist()
        }
    
    def getBoxAspectRatioDistribution(self, query = None):
        """
            return label_aspect_ratio, predict_aspect_ratio distribution
        """
        if query is None and self.box_aspect_ratio_dist_map is not None:
            return self.box_aspect_ratio_dist_map
        filtered, unmatch_predict, unmatch_label = self.filterSamples(query)
        iou_thres = self.iou_thresholds[0]
        if query is not None and "iou_thres" in query:
            iou_thres = query["iou_thres"]
        predict_label_pairs, _ = self.pairs_map_under_iou_thresholds[iou_thres]
        filtered_label_aspect_ratio = self.label_aspect_ratio[predict_label_pairs[filtered,1]]
        filtered_predict_aspect_ratio = self.predict_aspect_ratio[predict_label_pairs[filtered,0]]
        unmatched_label_aspect_ratio = self.label_aspect_ratio[unmatch_label]
        unmatched_predict_aspect_ratio = self.predict_aspect_ratio[predict_label_pairs[unmatch_predict,0]]
        
        K = 25
        if query is not None and 'label_range' in query:
            label_range = query['label_range']
            predict_range = query['predict_range']
        else:
            maximum_val = np.max(np.concatenate((self.label_aspect_ratio, filtered_predict_aspect_ratio, unmatched_predict_aspect_ratio)))
            label_range = [0, maximum_val]
            predict_range = [0, maximum_val]
        label_split = np.array([label_range[0]+i*(label_range[1]-label_range[0])/K for i in range(K+1)])
        predict_split = np.array([predict_range[0]+i*(predict_range[1]-predict_range[0])/K for i in range(K+1)])
        
        label_box_aspect_ratio_dist, predict_box_aspect_ratio_dist = [0 for _ in range(K)], [0 for _ in range(K)]
        for i in range(K):
            label_box_aspect_ratio_dist[i] = np.count_nonzero(np.logical_and(filtered_label_aspect_ratio>label_split[i], filtered_label_aspect_ratio<=label_split[i+1])) + \
             np.count_nonzero(np.logical_and(unmatched_label_aspect_ratio>label_split[i], unmatched_label_aspect_ratio<=label_split[i+1]))
            predict_box_aspect_ratio_dist[i] = np.count_nonzero(np.logical_and(filtered_predict_aspect_ratio>predict_split[i], filtered_predict_aspect_ratio<=predict_split[i+1])) + \
             np.count_nonzero(np.logical_and(unmatched_predict_aspect_ratio>predict_split[i], unmatched_predict_aspect_ratio<=predict_split[i+1]))
            
        label_target, pred_target = np.arange(len(self.classID2Idx)-1), np.arange(len(self.classID2Idx)-1)
        label_box_aspect_ratio_confusion = [[[0 for _ in range(K)] for _ in range(len(pred_target)+1)] for _ in range(len(label_target)+1)]
        predict_box_aspect_ratio_confusion = [[[0 for _ in range(K)] for _ in range(len(pred_target)+1)] for _ in range(len(label_target)+1)]
        for (pr, gt) in predict_label_pairs[filtered]:
            if label_range[0] <= self.label_aspect_ratio[gt] <= label_range[1]:
                label_box_aspect_ratio_confusion[int(self.raw_labels[gt, 0])][int(self.raw_predicts[pr, 0])][bisect.bisect_left(label_split[1:-1], self.label_aspect_ratio[gt])]+=1
            if predict_range[0] <= self.predict_aspect_ratio[pr] <= predict_range[1]:
                predict_box_aspect_ratio_confusion[int(self.raw_labels[gt, 0])][int(self.raw_predicts[pr, 0])][bisect.bisect_left(predict_split[1:-1], self.predict_aspect_ratio[pr])]+=1
        for gt in unmatch_label:
            if label_range[0] <= self.label_aspect_ratio[gt] <= label_range[1]:
                label_box_aspect_ratio_confusion[int(self.raw_labels[gt, 0])][len(pred_target)][bisect.bisect_left(label_split[1:-1], self.label_aspect_ratio[gt])]+=1
        for (pr, gt) in predict_label_pairs[unmatch_predict]:
            if predict_range[0] <= self.predict_aspect_ratio[pr] <= predict_range[1]:
                predict_box_aspect_ratio_confusion[len(label_target)][int(self.raw_predicts[pr, 0])][bisect.bisect_left(predict_split[1:-1], self.predict_aspect_ratio[pr])]+=1
                
        return {
            'labelAspectRatioAll': label_box_aspect_ratio_dist,
            'predictAspectRatioAll': predict_box_aspect_ratio_dist,
            'labelAspectRatioConfusion': label_box_aspect_ratio_confusion,
            'predictAspectRatioConfusion': predict_box_aspect_ratio_confusion,
            'labelSplit': label_split.tolist(),
            'predictSplit': predict_split.tolist()
        }        
    
    def transformBottomLabelToTop(self, topLabels):
        topLabelChildren = {}
        topLabelSet = set(topLabels)
        def dfs(nodes):
            childrens = []
            for root in nodes:
                if type(root)==str:
                    childrens.append(root)
                    if root in topLabelSet:
                        topLabelChildren[root] = [root]
                else:
                    rootChildren = dfs(root['children'])
                    childrens += rootChildren
                    if root['name'] in topLabelSet:
                        topLabelChildren[root['name']] = rootChildren
            return childrens
        dfs(self.hierarchy)
        childToTop = {}
        for topLabelIdx in range(len(topLabels)):
            for child in topLabelChildren[topLabels[topLabelIdx]]:
                childToTop[child] = topLabelIdx
        n = len(self.names)
        labelTransform = np.zeros(n, dtype=int)
        for i in range(n):
            if not childToTop.__contains__(self.names[i]):
                print('not include ' + self.names[i])
            else:
                labelTransform[i] = childToTop[self.names[i]]
        return labelTransform.astype(int)
        
    def gridZoomIn(self, nodes, constraints, depth, aspectRatio, zoomin, iou_thres, conf_thres):
        predict_label_pairs, _ = self.pairs_map_under_iou_thresholds[iou_thres]
        filtered, unmatch_predict, unmatch_label = self.filterSamples({
            "iou_thres": iou_thres,
            "conf_thres": conf_thres
        })
        all_pred_ids = predict_label_pairs[np.concatenate((filtered, unmatch_predict)), 0]
        unmatch_label_ids = unmatch_label + len(self.raw_predicts)

        allfeatures = np.concatenate((self.pr_features, self.gt_features))
        neighbors = self.sampler.zoomin(nodes, 250, allfeatures)

        # filter after sampling
        neighbors = np.array(neighbors)[np.logical_or(np.isin(neighbors, all_pred_ids),
                                                      np.isin(neighbors, unmatch_label_ids))].tolist()


        zoomInConstraints = None
        zoomInConstraintX = None
        if constraints is not None:
            zoomInConstraints = []
            zoomInConstraintX = []
        zoomInNodes = neighbors
        if constraints is not None:
            zoomInConstraints = np.array(constraints)
            nodesset = set(neighbors)
            for node in nodes:
                if node in nodesset:
                    zoomInConstraintX.append(node)
            zoomInConstraintX = allfeatures[nodes]
        
        zoomInLabels, zoomInPreds, zoomInConfidence = [], [], []
        for node in zoomInNodes:
            if node < len(self.raw_predicts):
                if predict_label_pairs[predict_label_pairs[:,0]==node][0, 1] == -1:
                    zoomInLabels.append(-1)
                else:
                    zoomInLabels.append(self.raw_labels[predict_label_pairs[predict_label_pairs[:,0]==node][0, 1], 0])
                zoomInPreds.append(self.raw_predicts[node, 0])
                zoomInConfidence.append(self.raw_predicts[node, 1])
            else:
                zoomInLabels.append(self.raw_labels[node-len(self.raw_predicts), 0])
                zoomInPreds.append(-1)
                zoomInConfidence.append(0)
        zoomInLabels, zoomInPreds, zoomInConfidence = np.array(zoomInLabels, dtype=np.int32), np.array(zoomInPreds, dtype=np.int32), np.array(zoomInConfidence, dtype=np.float64)
        zoomInLabels[zoomInLabels==-1] = len(self.names) - 1
        zoomInPreds[zoomInPreds==-1] = len(self.names) - 1

        def getBottomLabels(zoomInNodes):
            hierarchy = copy.deepcopy(self.hierarchy)
            labelnames = copy.deepcopy(self.names)
            nodes = [{
                "index": zoomInNodes[i],
                "label": zoomInLabels[i],
                "pred": zoomInPreds[i]
            } for i in range(len(zoomInNodes))]

            root = {
                'name': '',
                'children': hierarchy,
            }
            counts = {}
            for node in nodes:
                if not counts.__contains__(labelnames[node['pred']]):
                    counts[labelnames[node['pred']]] = 0
                counts[labelnames[node['pred']]] += 1

            def dfsCount(root, counts):
                if isinstance(root, str):
                    if not counts.__contains__(root): # todo
                        counts[root] = 0
                    return {
                        'name': root,
                        'count': counts[root],
                        'children': [],
                        'realChildren': [],
                        'emptyChildren': [],
                    }
                else:
                    count = 0
                    realChildren = []
                    emptyChildren = []
                    for i in range(len(root['children'])):
                        root['children'][i] = dfsCount(root['children'][i], counts)
                        count += root['children'][i]['count']
                        if root['children'][i]['count'] != 0:
                            realChildren.append(root['children'][i])
                        else: 
                            emptyChildren.append(root['children'][i])
                    root['realChildren'] = realChildren
                    root['emptyChildren'] = emptyChildren
                    counts[root['name']] = count
                    root['count'] = count
                    return root
            
            dfsCount(root, counts)

            pq = PriorityQueue()

            class Cmp:
                def __init__(self, name, count, realChildren):
                    self.name = name
                    self.count = count
                    self.realChildren = realChildren

                def __lt__(self, other):
                    if self.count <= other.count:
                        return False
                    else:
                        return True

                def to_list(self):
                    return [self.name, self.count, self.realChildren]
            
            pq.put(Cmp(root['name'], root['count'], root['realChildren']))
            classThreshold = 10
            countThreshold = 0.5
        
            while True:
                if pq.qsize()==0:
                    break
                top = pq.get()
                if pq.qsize() + len(top.realChildren) <= classThreshold or top.count / root['count'] >= countThreshold:
                    for child in top.realChildren:
                        pq.put(Cmp(child['name'], child['count'], child['realChildren']))
                    if pq.qsize()==0:
                        pq.put(top)
                        break
                else:
                    pq.put(top)
                    break
    
            pq_list = []
            while not pq.empty():
                pq_list.append(pq.get().name)
            return pq_list

        bottomLabels = getBottomLabels(copy.deepcopy(zoomInNodes))
        labelTransform = self.transformBottomLabelToTop(bottomLabels)
        constraintLabels = []
        for node in nodes:
            if node < len(self.raw_predicts):
                if predict_label_pairs[predict_label_pairs[:,0]==node][0, 1] == -1:
                    constraintLabels.append(labelTransform[len(self.names) - 1])
                else:
                    constraintLabels.append(labelTransform[int(self.raw_labels[predict_label_pairs[predict_label_pairs[:,0]==node][0, 1], 0])])
            else:
                constraintLabels.append(labelTransform[int(self.raw_labels[node-len(self.raw_predicts), 0])])
        labels = labelTransform[zoomInLabels]

        # oroginal
        # labelTransform = self.transformBottomLabelToTop([node['name'] for node in self.statistic['confusion']['hierarchy']])       

        tsne, grid, grid_width, grid_height = self.grider.fit(allfeatures[zoomInNodes], labels = labels, constraintX = zoomInConstraintX, 
                                               constraintY = zoomInConstraints, constraintLabels = constraintLabels, aspectRatio = aspectRatio)
        tsne = tsne.tolist()
        grid = grid.tolist()
        zoomInLabels = zoomInLabels.tolist()
        zoomInPreds = zoomInPreds.tolist()
        zoomInConfidence = zoomInConfidence.tolist()

        n = len(zoomInNodes)
        nodes = [{
            "index": zoomInNodes[i],
            "tsne": tsne[i],
            "grid": grid[i],
            "label": zoomInLabels[i],
            "pred": zoomInPreds[i],
            "confidence": zoomInConfidence[i]
        } for i in range(n)]
        res = {
            "nodes": nodes,
            "grid": {
                "width": grid_width,
                "height": grid_height,
            },
            "depth": 0
        }
        return res
        
    def pairIDtoImageID(self, boxID: int):
        if boxID>=len(self.raw_predicts):
            return self.raw_label2imageid[boxID-len(self.raw_predicts)]
        return self.raw_predict2imageid[boxID]
    
    def _getBoxesByImgId(self, img_id: int, iou_thres: float, conf_thres: float):
        predict_label_pairs, _ = self.pairs_map_under_iou_thresholds[iou_thres]
        # as 1 gt may occur in many pairs, so pr_boxes will only contain predict indexes, and so as gt_boxes
        pr_boxes = predict_label_pairs[np.logical_and(np.logical_and(predict_label_pairs[:, 0]>=self.imageid2raw_predict[img_id][0],
                                                                     predict_label_pairs[:, 0]< self.imageid2raw_predict[img_id][1]), 
                                                      self.raw_predicts[predict_label_pairs[:, 0], 1] > conf_thres), 0]
        gt_boxes = np.arange(self.imageid2raw_label[img_id][0], self.imageid2raw_label[img_id][1])
        return pr_boxes.tolist(), gt_boxes.tolist()

    def _getBoxByBoxId(self, box_id: int, iou_thres: float):
        predict_label_pairs, _ = self.pairs_map_under_iou_thresholds[iou_thres]
        pr_box, gt_box = [], []
        if box_id >= len(self.raw_predicts):
            pr_box.append(-1)
            gt_box.append(box_id - len(self.raw_predicts))
        else:
            pr_box.append(box_id)
            gt_box.append(int(predict_label_pairs[predict_label_pairs[:, 0] == box_id][0, 1]))
        return pr_box, gt_box
        
    def getImagebox(self, boxID: int, showall: str, iou_thres: float, conf_thres: float):
        finalBoxes = []
        img = Image.open(os.path.join(self.images_path, self.index2image[self.pairIDtoImageID(boxID)]+'.jpg'))
        amp = [img.width,img.height]
        if showall == 'all':
            imgID = self.pairIDtoImageID(boxID)
            pr_boxes, gt_boxes = self._getBoxesByImgId(imgID, iou_thres, conf_thres)
        elif showall == 'single':
            pr_boxes, gt_boxes = self._getBoxByBoxId(boxID, iou_thres)
        for box in pr_boxes:
            if box != -1:
                finalBoxes.append({
                    "box": (self.raw_predicts[box, 2:6]).tolist(),
                    "size": float(self.predict_size[box]),
                    "type": "pred",
                    "class": self.names[int(self.raw_predicts[box, 0])],
                    "id": box,
                    "score": float(self.raw_predicts[box, 1])
                })
        for box in gt_boxes:
            if box != -1:
                finalBoxes.append({
                    "box": (self.raw_labels[box, 1:5]).tolist(),
                    "size": float(self.label_size[box]),
                    "type": "gt",
                    "class": self.names[int(self.raw_labels[box, 0])],
                    "id": box,
                    "score": 0
                })
        return {
            "boxes": finalBoxes,
            "image": amp
        }
        
        
    def getImage(self, boxID: int, show: str, showall: str, iou_thres: float, conf_thres: float, hideBox = False):
        img = Image.open(os.path.join(self.images_path, self.index2image[self.pairIDtoImageID(boxID)]+'.jpg'))
        anno = Annotator(np.array(img), pil=True)
        amp = np.array([img.width,img.height,img.width,img.height])
        if showall == 'all':
            imgID = self.pairIDtoImageID(boxID)
            pr_boxes, gt_boxes = self._getBoxesByImgId(imgID, iou_thres, conf_thres)
        elif showall == 'single':
            pr_boxes, gt_boxes = self._getBoxByBoxId(boxID, iou_thres)
        for box in pr_boxes:
            predictXYXY = None
            if box != -1:
                predictXYXY = xywh2xyxy(self.raw_predicts[box, 2:6]*amp).tolist()
                anno.box_label(predictXYXY, color=(255, 0, 0))
        for box in gt_boxes:
            labelXYXY = None
            if box != -1:
                labelXYXY = xywh2xyxy(self.raw_labels[box, 1:5]*amp).tolist()
                anno.box_label(labelXYXY, color=(0, 255, 0))
        output = io.BytesIO()
        if hideBox:
            img.save(output, format="JPEG")
        elif show=='box':
            self.cropImageByBox(anno.im, predictXYXY, labelXYXY, [img.width, img.height]).save(output, format="JPEG")
        else:
            anno.im.save(output, format="JPEG")
        return output
    
    def getImages(self, boxIDs: list, show: str, iou_thres: float):
        base64Imgs = []
        for boxID in boxIDs:
            img = Image.open(os.path.join(self.images_path, self.index2image[self.pairIDtoImageID(boxID)]+'.jpg'))
            anno = Annotator(np.array(img), pil=True)
            amp = np.array([img.width,img.height,img.width,img.height])
            pr_boxes, gt_boxes = self._getBoxByBoxId(boxID, iou_thres)
            predictBox, labelBox = pr_boxes[0], gt_boxes[0]
            predictXYXY = None
            if predictBox != -1:
                predictXYXY = xywh2xyxy(self.raw_predicts[predictBox, 2:6]*amp).tolist()
                anno.box_label(predictXYXY, color=(255,102,0))
            labelXYXY = None
            if labelBox != -1:
                labelXYXY = xywh2xyxy(self.raw_labels[labelBox, 1:5]*amp).tolist()
                anno.box_label(labelXYXY, color=(95,198,181))
            output = io.BytesIO()
            if show=='box':
                self.cropImageByBox(anno.im, predictXYXY, labelXYXY, [img.width, img.height]).save(output, format="JPEG")
            else:
                anno.im.save(output, format="JPEG")
            base64Imgs.append(base64.b64encode(output.getvalue()).decode())
        return base64Imgs
            
    def cropImageByBox(self, img, predictBox, labelBox, shape):
        box = None
        if predictBox is None:
            box = labelBox
        elif labelBox is None:
            box = predictBox
        else:        
            box = [
                min(predictBox[0], labelBox[0]),
                min(predictBox[1], labelBox[1]),
                max(predictBox[2], labelBox[2]),
                max(predictBox[3], labelBox[3])
            ]
        center = [(box[0]+box[2])/2, (box[1]+box[3])/2]
        size = max((box[2]-box[0])/2, (box[3]-box[1])/2)
        if (box[2]-box[0])*(box[3]-box[1])<400:
            size += 10
        box = [
            max(center[0]-size, 0),
            max(center[1]-size, 0),
            min(center[0]+size, shape[0]),
            min(center[1]+size, shape[1])
        ]
        return img.crop(box)
        
    def getImagesInConsuionMatrixCell(self, labels: list, preds: list, query = None) -> list:
        """
        return images in a cell of confusionmatrix

        Args:
            labels (list): true labels of corresponding cell
            preds (list): predicted labels of corresponding cell

        Returns:
            list: images id
        """ 
        iou_thres = self.iou_thresholds[0]
        if query is not None and "iou_thres" in query:
            iou_thres = query["iou_thres"]
        predict_label_pairs, _ = self.pairs_map_under_iou_thresholds[iou_thres]
        # convert list of label names to dict
        labelNames = self.names
        name2idx = {}
        for i in range(len(labelNames)):
            name2idx[labelNames[i]]=i
        # find images
        labelSet = set()
        for label in labels:
            labelSet.add(name2idx[label])
        predSet = set()
        for label in preds:
            predSet.add(name2idx[label])
        query["label"] = list(labelSet)
        query["predict"] = list(predSet)
        filtered , unmatch_predict, unmatch_label = self.filterSamples(query)
        imageids = []
        if (len(self.names)-1) in labelSet and (len(self.names)-1) in predSet:
            pass
        elif (len(self.names)-1) in labelSet:
            imageids = predict_label_pairs[unmatch_predict, 0].tolist()
        elif (len(self.names)-1) in predSet:
            imageids = (unmatch_label + len(self.raw_predicts)).tolist()
        else:
            imageids = predict_label_pairs[filtered, 0].tolist()
        # limit length of images
        return imageids
    
    def getClassStatistics(self, query = None):
        """
            area: [0, 1, 2, 3] => [all, small, medium, large]
            mdets: [0, 1, 2] => [1, 10, 100]
            ap: 1 / 0 => precision / recall
            iouThr: None / 0.50 / 0.75
        """
        iouThrs = np.array([i*0.05+0.5 for i in range(10)])
        default_query = {
            "area": 0,
            "maxDets": 2,
            "ap": 1
        }
        if query is not None:
            query = {**default_query, **query}
        area = query['area']
        mDets = query['maxDets']
        ap = query['ap']
        if 'iouThr' in query:
            iouThr = query['iouThr']
        else:
            iouThr = None
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval_data['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == iouThrs)[0]
                s = s[t]
            s = s[:,:,:,area,mDets]
        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval_data['recall']
            if iouThr is not None:
                t = np.where(iouThr == iouThrs)[0]
                s = s[t]
            s = s[:,:,area,mDets]
        # mean_s = np.mean(s[s>-1])
        # cacluate AP/AR for all categories
        if ap == 1:
            return [float(np.mean(s[:,:,i][s[:,:,i]>-1])) for i in range(80)]
        else:
            return [float(np.mean(s[:,i][s[:,i]>-1])) for i in range(80)]
    
def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    if len(x.shape)==1:
        y[0] = x[0] - x[2] / 2  # top left x
        y[1] = x[1] - x[3] / 2  # top left y
        y[2] = x[0] + x[2] / 2  # bottom right x
        y[3] = x[1] + x[3] / 2  # bottom right y
        return y
    else:
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

def cal_iou(pr, gt, iscrowd = None):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
            Return intersection-over-union (Jaccard index) of boxes.
            Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
            Arguments:
                pr (Tensor[N, 4])
                gt (Tensor[M, 4])
                iscrowd ([M])
            Returns:
                iou (Tensor[N, M]): the NxM matrix containing the pairwise
                    IoU values for every element in boxes1 and boxes2
        """
        pr, gt = xywh2xyxy(pr), xywh2xyxy(gt)
        (a1, a2), (b1, b2) = np.array_split(pr[:, None],2,axis=2), np.array_split(gt, 2, axis=1)
        inter = (np.minimum(a2, b2) - np.maximum(a1, b1)).clip(0).prod(2)
        union = (box_area(pr.T)[:, None] + box_area(gt.T) - inter)
        if iscrowd is not None:
            crowd_idx = np.where(iscrowd == 1)[0]
            if len(crowd_idx) > 0:
                union[:, crowd_idx] = box_area(pr.T)[:, None]
        return inter / (union + 1e-6)

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = np.array_split(box1[:, None],2,axis=2), np.array_split(box2, 2, axis=1)
    inter = (np.minimum(a2, b2) - np.maximum(a1, b1)).clip(0).prod(2)
    union = (box_area(box1.T)[:, None] + box_area(box2.T) - inter)
    # IoU = inter / (area1 + area2 - inter)
    return inter / (union + 1e-6), union

def generalized_box_iou(box1, box2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    box1, box2 = xywh2xyxy(box1), xywh2xyxy(box2)
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (box1[:, 2:] >= box1[:, :2]).all()
    assert (box2[:, 2:] >= box2[:, :2]).all()

    iou, union = box_iou(box1, box2)
    box1, box2 = torch.from_numpy(box1), torch.from_numpy(box2)

    lt = torch.min(box1[:, None, :2], box2[:, :2])
    rb = torch.max(box1[:, None, 2:], box2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = np.array(wh[:, :, 0] * wh[:, :, 1])

    return iou - (area - union) / (area + 1e-6)

dataCtrler = DataCtrler()

if __name__ == "__main__":
    dataCtrler.process("/data/yukai/UnifiedConfusionMatrix/datasets/coco/", "/data/yukai/UnifiedConfusionMatrix/buffer/")
    # dataCtrler.process("/data/zhaowei/ConfusionMatrix/datasets/coco/", "/data/zhaowei/ConfusionMatrix/backend/buffer/")
    matrix = dataCtrler.getConfusionMatrix({
        "label_size": [0,1],
        "predict_size": [0,1],
        "label_aspect_ratio": [0,1],
        "predict_aspect_ratio": [0,1],
        "direction": [0,1,2,3,4,5,6,7,8],
        "label": np.arange(80),
        "predict": np.arange(80),
        "return": ['size_comparison'],
        "split": 10
    })