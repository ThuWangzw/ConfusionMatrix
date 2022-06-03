import os
import json
import copy
import bisect
from tkinter import N
import numpy as np
# import data.reorder as reorder
import pickle
import torch
import math
from PIL import Image
import logging
from queue import PriorityQueue

from data.grid.sampling import HierarchySampling
from data.grid.gridLayout import GridLayout
from data.fisher import get_split_pos
from data.drawBox import Annotator

class DataCtrler(object):

    def __init__(self):
        super().__init__()
        self.iou_threshold_localization = 0.5
        self.iou_threshold_miss = 0.1
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
        self.features_path = os.path.join(self.root_path, "features")
        if not os.path.exists(self.features_path):
            os.makedirs(self.features_path)
        if not os.path.exists(bufferPath):
            os.makedirs(bufferPath)
        self.raw_data_path = os.path.join(bufferPath, "{}_raw_data.pkl".format(os.path.basename(os.path.normpath(rawDataPath))))
        self.label_predict_iou_path = os.path.join(bufferPath, "{}_predict_label_iou.pkl".format(os.path.basename(os.path.normpath(rawDataPath))))
        self.box_size_split_path = os.path.join(bufferPath, "{}_box_size_split.pkl".format(os.path.basename(os.path.normpath(rawDataPath))))
        self.box_size_dist_path = os.path.join(bufferPath, "{}_box_size_dist.pkl".format(os.path.basename(os.path.normpath(rawDataPath))))
        self.box_aspect_ratio_split_path = os.path.join(bufferPath, "{}_box_aspect_ratio_split.pkl".format(os.path.basename(os.path.normpath(rawDataPath))))
        self.box_aspect_ratio_dist_path = os.path.join(bufferPath, "{}_box_aspect_ratio_dist.pkl".format(os.path.basename(os.path.normpath(rawDataPath))))
        self.hierarchy_sample_path = os.path.join(bufferPath, "{}_hierarchy_samples.pkl".format(os.path.basename(os.path.normpath(rawDataPath))))
        self.all_features_path = os.path.join(bufferPath, "{}_features.npy".format(os.path.basename(os.path.normpath(rawDataPath))))
        
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
            # format: label, box(cx, cy, w, h)
            self.raw_labels = np.zeros((0,5), dtype=np.float32)
            self.raw_label2imageid = np.zeros(0, dtype=np.int32)
            self.imageid2raw_label = np.zeros((id, 2), dtype=np.int32)
            for imageName in os.listdir(self.labels_path):
                label_path = os.path.join(self.labels_path, imageName)
                imageid = self.image2index[imageName.split('.')[0]]
                with open(label_path) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any([len(x)>5 for x in lb]):
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
        
        # read feature data
        if os.path.exists(self.all_features_path):
            self.features = np.load(self.all_features_path)
        else:
            self.features = np.zeros((self.raw_predicts.shape[0], 256))
            for name in os.listdir(self.images_path):
                feature_path = os.path.join(self.features_path, name.split('.')[0]+'.npy')
                imageid = self.image2index[name.split('.')[0]]
                boxCount = self.imageid2raw_predict[imageid][1]-self.imageid2raw_predict[imageid][0]
                if not os.path.exists(feature_path):
                    # WARNING
                    self.logger.warning("can't find feature: %s" % feature_path)
                    self.features[self.imageid2raw_predict[imageid][0]:self.imageid2raw_predict[imageid][1]] = np.random.rand(boxCount, 256)
                else:
                    self.features[self.imageid2raw_predict[imageid][0]:self.imageid2raw_predict[imageid][1]] = np.load(feature_path)
            np.save(self.all_features_path, self.features)
        
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
        if os.path.exists(self.label_predict_iou_path):
            with open(self.label_predict_iou_path, 'rb') as f:
                self.predict_label_pairs, self.predict_label_ious = pickle.load(f)
        else:
            self.predict_label_pairs, self.predict_label_ious = self.compute_label_predict_pair()
            with open(self.label_predict_iou_path, 'wb') as f:
                pickle.dump((self.predict_label_pairs, self.predict_label_ious), f)
        
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
        
        ## init size
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

        ## init aspect ratio
        self.label_aspect_ratio = self.raw_labels[:,3]/self.raw_labels[:,4]
        self.predict_aspect_ratio = self.raw_predicts[:,4]/self.raw_predicts[:,5]

        # get box aspect ratio distribution in [0,1]
        if os.path.exists(self.box_aspect_ratio_dist_path):
            with open(self.box_aspect_ratio_dist_path, 'rb') as f:
                self.box_aspect_ratio_dist_map = pickle.load(f)
        else:
            self.box_aspect_ratio_dist_map = None
            self.box_aspect_ratio_dist_map = self.getBoxAspectRatioDistribution()
            with open(self.box_aspect_ratio_dist_path, 'wb') as f:
                pickle.dump(self.box_aspect_ratio_dist_map, f)
        ## direction
        directionIdxes = np.where(np.logical_and(self.predict_label_pairs[:,0]>-1, self.predict_label_pairs[:,1]>-1))
        directionVectors = self.raw_predicts[self.predict_label_pairs[directionIdxes][:,0]][:,[2,3]] - self.raw_labels[self.predict_label_pairs[directionIdxes][:,1]][:,[1,2]]
        directionNorm = np.sqrt(np.power(directionVectors[:,0], 2)+ np.power(directionVectors[:,1], 2))
        directionCos = directionVectors[:,0]/directionNorm
        directions = np.zeros(directionCos.shape[0], dtype=np.int32)
        directionSplits = np.array([math.cos(angle/180*math.pi) for angle in [180, 157.5, 112.5, 67.5, 22.5, 0]])
        for i in range(2,len(directionSplits)):
            directions[np.logical_and(directionCos>directionSplits[i-1], directionCos<=directionSplits[i])] = i-1
        negaYs = np.logical_and(directionVectors[:,1]<0, directions!=0)
        directions[negaYs] = 8-directions[negaYs]
        directions[directionNorm<0.05] = 8
        self.directions = -1*np.ones(self.predict_label_pairs.shape[0], dtype=np.int32)
        self.directions[directionIdxes] = directions
        
        # hierarchy sampling
        self.sampler = HierarchySampling()
        if os.path.exists(self.hierarchy_sample_path):
            self.sampler.load(self.hierarchy_sample_path)
        else:
            labels = self.raw_predicts[:, 0].astype(np.int32)
            self.sampler.fit(self.features, labels, 0.5, 400)
            self.sampler.dump(self.hierarchy_sample_path)
          
    def getMetaData(self):
        return {
            "hierarchy": self.hierarchy,
            "names": self.names
        }
            
    def compute_label_predict_pair(self):
        def compute_per_image(detections, labels):
            # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
            iou = box_iou(detections[:, 2:6], labels[:, 1:5])

            x = np.where(iou > self.iou_threshold_miss)
            if x[0].shape[0]:
                matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            else:
                matches = np.zeros((0, 3))
            final_match = -1*np.ones((len(detections),2), dtype=np.int32)
            ious = np.zeros(len(detections))
            final_match[:,0] = np.arange(len(detections), dtype=np.int32)
            for match in matches:
                final_match[int(match[0])] = match[:2].astype(np.int32)
                ious[int(match[0])] = match[2]
            return final_match, ious
        self.predict_label_pairs = -1*np.ones((len(self.raw_predicts), 2), dtype=np.int32)
        self.predict_label_ious = np.zeros(len(self.raw_predicts))
        for imageidx in range(len(self.image2index)):
            matches, ious = compute_per_image(self.raw_predicts[self.imageid2raw_predict[imageidx][0]:self.imageid2raw_predict[imageidx][1]],
                                        self.raw_labels[self.imageid2raw_label[imageidx][0]:self.imageid2raw_label[imageidx][1]])
            negaWeights = np.where(matches[:,1]==-1)[0]
            if len(matches)>0:
                matches[:,1]+=self.imageid2raw_label[imageidx][0]
                matches[:,0]+=self.imageid2raw_predict[imageidx][0]
                matches[negaWeights,1]=-1
                self.predict_label_pairs[self.imageid2raw_predict[imageidx][0]:self.imageid2raw_predict[imageidx][1]] = matches
                self.predict_label_ious[self.imageid2raw_predict[imageidx][0]:self.imageid2raw_predict[imageidx][1]] = ious
        # for labels that haven't been matched to any predict
        label_not_match = np.setdiff1d(np.arange(len(self.raw_labels)), self.predict_label_pairs[:, 1])
        label_extra_pairs = -1 * np.ones((len(label_not_match), 2), dtype=np.int32)
        label_extra_pairs[:, 1] = label_not_match
        label_extra_ious = np.zeros(len(label_not_match))
        self.predict_label_pairs = np.concatenate((self.predict_label_pairs, label_extra_pairs))
        self.predict_label_ious = np.concatenate((self.predict_label_ious, label_extra_ious))
        return self.predict_label_pairs, self.predict_label_ious
    
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
            "split": 10
        }
        if query is not None:
            query = {**default_query, **query}
        # separate matched pairs from unmatch ones
        # index of pred_label_pairs
        filtered = np.arange(len(self.predict_label_pairs))[np.logical_and(self.predict_label_pairs[:,1]>-1, self.predict_label_pairs[:,0]>-1)]
        unmatch_predict = np.arange(len(self.predict_label_pairs))[self.predict_label_pairs[:,1]==-1]
        unmatch_label = np.arange(len(self.predict_label_pairs))[self.predict_label_pairs[:,0]==-1]

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
            filtered = filtered[np.isin(self.predict_label_pairs[filtered][:,1], label_selected)]
            filtered = filtered[np.isin(self.predict_label_pairs[filtered][:,0], predict_selected)]
            unmatch_predict = unmatch_predict[np.isin(self.predict_label_pairs[unmatch_predict][:,0], predict_selected)]
            unmatch_label = unmatch_label[np.isin(self.predict_label_pairs[unmatch_label][:,1], label_selected)]

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
    
    def getStatisticsMatrixes(self, matrix, query):
        """
            matrix: a 3-d list consists of lists of indexes from predict_label_pairs 
        """
        statistics_modes = ['count']
        if query is not None and "return" in query:
            statistics_modes = query['return']
        function_map = {
            'count': lambda x: len(x),
            'avg_label_size': lambda x: 0 if self.predict_label_pairs[x[0],1]==-1 else self.label_size[self.predict_label_pairs[x, 1]].mean(),
            'avg_predict_size': lambda x: 0 if self.predict_label_pairs[x[0],0]==-1 else self.predict_size[self.predict_label_pairs[x, 0]].mean(),
            'avg_iou': lambda x: self.predict_label_ious[x].mean(),
            'avg_acc': lambda x: 0 if np.any(self.predict_label_pairs[x[0],:]==-1) else 
                            (self.raw_predicts[self.predict_label_pairs[x,0], 0]==self.raw_labels[self.predict_label_pairs[x,1], 0]).mean(),
            'avg_label_aspect_ratio': lambda x: 0 if self.predict_label_pairs[x[0],1]==-1 else self.label_aspect_ratio[self.predict_label_pairs[x, 1]].mean(),
            'avg_predict_aspect_ratio': lambda x: 0 if self.predict_label_pairs[x[0],0]==-1 else self.predict_aspect_ratio[self.predict_label_pairs[x, 0]].mean(),
            'direction': lambda x: [int(np.count_nonzero(self.directions[x]==i)) for i in range(9)]
        }
        ret_matrixes = []
        for statistics_mode in statistics_modes:
            if statistics_mode not in function_map:
                raise NotImplementedError()
            map_func = function_map[statistics_mode]
            stat_matrix = np.zeros((len(matrix), len(matrix[0])), dtype=np.float64).tolist()
            for i in range(len(stat_matrix)):
                for j in range(len(stat_matrix[0])):
                    if statistics_mode != 'direction' and len(matrix[i][j]) == 0:
                        continue
                    if statistics_mode == 'direction' and (i == len(stat_matrix)-1 or j == len(stat_matrix[0])-1):
                        stat_matrix[i][j] = [0 for _ in range(9)]
                    else:
                        stat_matrix[i][j] = map_func(matrix[i][j])
            ret_matrixes.append(np.array(stat_matrix).tolist())
        return ret_matrixes

    def getConfusionMatrix(self, query = None):
        """filtered confusion matrix

        Args:
            querys (dict): {label/predict size:[a, b], label/predict aspect_ratio:[a, b], direction: [0,..,8],
            label/predict: np.arange(80)}
        """
        filtered , unmatch_predict, unmatch_label = self.filterSamples(query)
        label_target, pred_target = np.arange(80), np.arange(80)
        if query is not None and "label" in query and "predict" in query:
            label_target = query["label"]
            pred_target = query["predict"]
        confusion = [[[] for _ in range(len(pred_target)+1)] for _ in range(len(label_target)+1)]
        label_rec, pred_rec = [], []
        for i in range(len(label_target)):
            label_rec.append(self.raw_labels[self.predict_label_pairs[filtered][:, 1], 0]==label_target[i])
        for j in range(len(pred_target)):
            pred_rec.append(self.raw_predicts[self.predict_label_pairs[filtered][:, 0], 0]==pred_target[j])
        for i in range(len(label_target)):
            for j in range(len(pred_target)):
                confusion[i][j] = filtered[np.logical_and(label_rec[i], pred_rec[j])]
        for i in range(len(label_target)):
            confusion[i][len(pred_target)] = unmatch_label[self.raw_labels[self.predict_label_pairs[unmatch_label][:, 1], 0]==label_target[i]]
        for j in range(len(pred_target)):
            confusion[len(label_target)][j] = unmatch_predict[self.raw_predicts[self.predict_label_pairs[unmatch_predict][:, 0], 0]==pred_target[j]]
        return self.getStatisticsMatrixes(confusion, query)
   
    def getBoxSizeDistribution(self, query = None):
        """
            return label_size, pred_size distribution
        """
        if query is None and self.box_size_dist_map is not None:
            return self.box_size_dist_map
        filtered, unmatch_predict, unmatch_label = self.filterSamples(query)
        filtered_label_size = self.label_size[self.predict_label_pairs[filtered][:,1]]
        filtered_predict_size = self.predict_size[self.predict_label_pairs[filtered][:,0]]
        unmatched_label_size = self.label_size[self.predict_label_pairs[unmatch_label][:,1]]
        unmatched_predict_size = self.predict_size[self.predict_label_pairs[unmatch_predict][:,0]]
        
        # partition
        K = 25
        # pred_size_argsort = np.argsort(self.predict_size)
        # if K not in self.box_size_split_map:
        #     self.box_size_split_map[K] = get_split_pos(self.predict_label_ious[pred_size_argsort], K)
        #     with open(self.box_size_split_path, 'wb') as f:
        #         pickle.dump(self.box_size_split_map, f)
        # split_pos = self.box_size_split_map[K]
        # split_size = self.predict_size[pred_size_argsort][split_pos]
        maximum_val = np.max(np.concatenate((self.label_size, self.predict_size)))
        split_size = np.array([i*maximum_val/K for i in range(K+1)])
        
        label_box_size_dist, predict_box_size_dist = [0 for _ in range(K)], [0 for _ in range(K)]
        for i in range(K):
            last = split_size[i]
            cur = split_size[i+1]
            label_box_size_dist[i] = np.count_nonzero(np.logical_and(filtered_label_size>last, filtered_label_size<=cur)) + \
             np.count_nonzero(np.logical_and(unmatched_label_size>last, unmatched_label_size<=cur))
            predict_box_size_dist[i] = np.count_nonzero(np.logical_and(filtered_predict_size>last, filtered_predict_size<=cur)) + \
             np.count_nonzero(np.logical_and(unmatched_predict_size>last, unmatched_predict_size<=cur))
            
        label_target, pred_target = np.arange(len(self.classID2Idx)-1), np.arange(len(self.classID2Idx)-1)
        label_box_size_confusion = [[[0 for _ in range(K)] for _ in range(len(pred_target)+1)] for _ in range(len(label_target)+1)]
        predict_box_size_confusion = [[[0 for _ in range(K)] for _ in range(len(pred_target)+1)] for _ in range(len(label_target)+1)]
        for (pr, gt) in self.predict_label_pairs[filtered]:
            label_box_size_confusion[int(self.raw_labels[gt, 0])][int(self.raw_predicts[pr, 0])][bisect.bisect_left(split_size[1:-1], self.label_size[gt])]+=1
            predict_box_size_confusion[int(self.raw_labels[gt, 0])][int(self.raw_predicts[pr, 0])][bisect.bisect_left(split_size[1:-1], self.predict_size[pr])]+=1
        for (pr, gt) in self.predict_label_pairs[unmatch_label]:
            label_box_size_confusion[int(self.raw_labels[gt, 0])][len(pred_target)][bisect.bisect_left(split_size[1:-1], self.label_size[gt])]+=1
        for (pr, gt) in self.predict_label_pairs[unmatch_predict]:
            predict_box_size_confusion[len(label_target)][int(self.raw_predicts[pr, 0])][bisect.bisect_left(split_size[1:-1], self.predict_size[pr])]+=1
                
        return {
            'labelSizeAll': label_box_size_dist,
            'predictSizeAll': predict_box_size_dist,
            'labelSizeConfusion': label_box_size_confusion,
            'predictSizeConfusion': predict_box_size_confusion,
            'sizeSplit': split_size.tolist()
        }
    
    def getBoxAspectRatioDistribution(self, query = None):
        """
            return label_aspect_ratio, predict_aspect_ratio distribution
        """
        if query is None and self.box_aspect_ratio_dist_map is not None:
            return self.box_aspect_ratio_dist_map
        filtered, unmatch_predict, unmatch_label = self.filterSamples(query)
        filtered_label_aspect_ratio = self.label_aspect_ratio[self.predict_label_pairs[filtered,1]]
        filtered_predict_aspect_ratio = self.predict_aspect_ratio[self.predict_label_pairs[filtered,0]]
        unmatched_label_aspect_ratio = self.label_aspect_ratio[self.predict_label_pairs[unmatch_label,1]]
        unmatched_predict_aspect_ratio = self.predict_aspect_ratio[self.predict_label_pairs[unmatch_predict,0]]
        
        # partition
        K = 25
        # pred_aspect_ratio_argsort = np.argsort(self.predict_aspect_ratio)
        # if K not in self.box_aspect_ratio_split_map:
        #     self.box_aspect_ratio_split_map[K] = get_split_pos(self.predict_label_ious[pred_aspect_ratio_argsort], K)
        #     with open(self.box_aspect_ratio_split_path, 'wb') as f:
        #         pickle.dump(self.box_aspect_ratio_split_map, f)
        # split_pos = self.box_aspect_ratio_split_map[K]
        # split_aspect_ratio = self.predict_aspect_ratio[pred_aspect_ratio_argsort][split_pos]
        maximum_val = np.max(np.concatenate((self.label_aspect_ratio, self.predict_aspect_ratio)))
        # split_aspect_ratio = np.array([i*maximum_val/K for i in range(K+1)])
        split_aspect_ratio = np.array([i*4.0/(K-1) for i in range(K)]+[maximum_val])
        
        label_box_aspect_ratio_dist, predict_box_aspect_ratio_dist = [0 for _ in range(K)], [0 for _ in range(K)]
        for i in range(K):
            last = split_aspect_ratio[i]
            cur = split_aspect_ratio[i+1]
            label_box_aspect_ratio_dist[i] = np.count_nonzero(np.logical_and(filtered_label_aspect_ratio>last, filtered_label_aspect_ratio<=cur)) + \
             np.count_nonzero(np.logical_and(unmatched_label_aspect_ratio>last, unmatched_label_aspect_ratio<=cur))
            predict_box_aspect_ratio_dist[i] = np.count_nonzero(np.logical_and(filtered_predict_aspect_ratio>last, filtered_predict_aspect_ratio<=cur)) + \
             np.count_nonzero(np.logical_and(unmatched_predict_aspect_ratio>last, unmatched_predict_aspect_ratio<=cur))
            
        label_target, pred_target = np.arange(len(self.classID2Idx)-1), np.arange(len(self.classID2Idx)-1)
        label_box_aspect_ratio_confusion = [[[0 for _ in range(K)] for _ in range(len(pred_target)+1)] for _ in range(len(label_target)+1)]
        predict_box_aspect_ratio_confusion = [[[0 for _ in range(K)] for _ in range(len(pred_target)+1)] for _ in range(len(label_target)+1)]
        for (pr, gt) in self.predict_label_pairs[filtered]:
            label_box_aspect_ratio_confusion[int(self.raw_labels[gt, 0])][int(self.raw_predicts[pr, 0])][bisect.bisect_left(split_aspect_ratio[1:-1], self.label_aspect_ratio[gt])]+=1
            predict_box_aspect_ratio_confusion[int(self.raw_labels[gt, 0])][int(self.raw_predicts[pr, 0])][bisect.bisect_left(split_aspect_ratio[1:-1], self.predict_aspect_ratio[pr])]+=1
        for (pr, gt) in self.predict_label_pairs[unmatch_label]:
            label_box_aspect_ratio_confusion[int(self.raw_labels[gt, 0])][len(pred_target)][bisect.bisect_left(split_aspect_ratio[1:-1], self.label_aspect_ratio[gt])]+=1
        for (pr, gt) in self.predict_label_pairs[unmatch_predict]:
            predict_box_aspect_ratio_confusion[len(label_target)][int(self.raw_predicts[pr, 0])][bisect.bisect_left(split_aspect_ratio[1:-1], self.predict_aspect_ratio[pr])]+=1
                
        return {
            'labelAspectRatioAll': label_box_aspect_ratio_dist,
            'predictAspectRatioAll': predict_box_aspect_ratio_dist,
            'labelAspectRatioConfusion': label_box_aspect_ratio_confusion,
            'predictAspectRatioConfusion': predict_box_aspect_ratio_confusion,
            'aspectRatioSplit': split_aspect_ratio.tolist()
        }        

    def findGridParent(self, children, parents):
        return self.sampler.findParents(children, parents)
    
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
        
    def gridZoomIn(self, nodes, constraints, depth):
        allpreds = self.raw_predicts[self.predict_label_pairs[:len(self.raw_predicts),0], 0].astype(np.int32)
        alllabels = self.raw_labels[self.predict_label_pairs[:len(self.raw_predicts),1], 0].astype(np.int32)
        negaLabels = np.where(self.predict_label_pairs[:len(self.raw_predicts),1]==-1)[0]
        alllabels[negaLabels] = len(self.names)-1
        allconfidence = self.raw_predicts[self.predict_label_pairs[:,0], 1]
        allfeatures = self.features
        neighbors, newDepth = self.sampler.zoomin(nodes, depth)
        if type(neighbors)==dict:
            while True:
                newnodes = []
                for neighbor in neighbors.values():
                    newnodes += neighbor
                if len(newnodes) >= 1000 or newDepth==self.sampler.max_depth:
                    break
                neighbors, newDepth = self.sampler.zoomin(newnodes, newDepth)
        zoomInConstraints = None
        zoomInConstraintX = None
        if constraints is not None:
            zoomInConstraints = []
            zoomInConstraintX = []
        zoomInNodes = []
        if type(neighbors)==list:
            zoomInNodes = neighbors
            if constraints is not None:
                zoomInConstraints = np.array(constraints)
                nodesset = set(neighbors)
                for node in nodes:
                    if node in nodesset:
                        zoomInConstraintX.append(node)
                zoomInConstraintX = allfeatures[nodes]
        else:
            for children in neighbors.values():
                for child in children:
                    if int(child) not in nodes:
                        zoomInNodes.append(int(child))
                    # initY.append([constraints[i][0], constraints[i][1]])
            if constraints is not None:
                zoomInConstraints = np.array(constraints)
                zoomInConstraintX = allfeatures[nodes]
            zoomInNodes = nodes + zoomInNodes
        zoomInLabels = alllabels[zoomInNodes]
        zoomInPreds = allpreds[zoomInNodes]
        zoomInConfidence = allconfidence[zoomInNodes]

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
        constraintLabels = labelTransform[alllabels[nodes]]
        labels = labelTransform[zoomInLabels]  

        # oroginal
        # labelTransform = self.transformBottomLabelToTop([node['name'] for node in self.statistic['confusion']['hierarchy']])       

        tsne, grid, gridsize = self.grider.fit(allfeatures[zoomInNodes], labels = labels, constraintX = zoomInConstraintX,  constraintY = zoomInConstraints, constraintLabels = constraintLabels)
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
                "width": gridsize,
                "height": gridsize,
            },
            "depth": newDepth
        }
        return res
        
    def getImage(self, boxID: int, show: str) -> list:
        import io
        img = Image.open(os.path.join(self.images_path, self.index2image[self.raw_predict2imageid[boxID]]+'.jpg'))
        anno = Annotator(np.array(img), pil=True)
        amp = np.array([img.width,img.height,img.width,img.height])
        predictBox, labelBox = self.predict_label_pairs[boxID]
        predictXYXY = None
        if predictBox != -1:
            predictXYXY = xywh2xyxy(self.raw_predicts[predictBox, 2:6]*amp).tolist()
            anno.box_label(predictXYXY, color=(255,0,0))
        labelXYXY = None
        if labelBox != -1:
            labelXYXY = xywh2xyxy(self.raw_labels[labelBox, 1:5]*amp).tolist()
            anno.box_label(labelXYXY, color=(0,255,0))
        output = io.BytesIO()
        if show=='box':
            self.cropImageByBox(anno.im, predictXYXY, labelXYXY, [img.width, img.height]).save(output, format="JPEG")
        else:
            anno.im.save(output, format="JPEG")
        return output
    
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
        
    def getImagesInConsuionMatrixCell(self, labels: list, preds: list) -> list:
        """
        return images in a cell of confusionmatrix

        Args:
            labels (list): true labels of corresponding cell
            preds (list): predicted labels of corresponding cell

        Returns:
            list: images id
        """ 
        allpreds = self.raw_predicts[self.predict_label_pairs[:len(self.raw_predicts),0], 0].astype(np.int32)
        alllabels = self.raw_labels[self.predict_label_pairs[:len(self.raw_predicts),1], 0].astype(np.int32)
        negaLabels = np.where(self.predict_label_pairs[:len(self.raw_predicts),1]==-1)[0]
        alllabels[negaLabels] = len(self.names)-1
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
        imageids = []
        if alllabels is not None and allpreds is not None:
            n = len(alllabels)
            for i in range(n):
                if alllabels[i] in labelSet and allpreds[i] in predSet:
                    imageids.append(i)
                    
        # limit length of images
        return imageids[:225]
    
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
    box1, box2 = xywh2xyxy(box1), xywh2xyxy(box2)
    (a1, a2), (b1, b2) = np.array_split(box1[:, None],2,axis=2), np.array_split(box2, 2, axis=1)
    inter = (np.minimum(a2, b2) - np.maximum(a1, b1)).clip(0).prod(2)
    
    # # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter)

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
        "return": ['count', 'direction'],
        "split": 10
    })