import os
import json
import numpy as np
# import data.reorder as reorder
import pickle
import torch

from data.fisher import get_size_split_pos

class DataCtrler(object):

    def __init__(self):
        super().__init__()
        self.iou_threshold_localization = 0.5
        self.iou_threshold_miss = 0.1
        self.classID2Idx = {}      
        self.hierarchy = {}  
        self.names = []

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
        if not os.path.exists(bufferPath):
            os.makedirs(bufferPath)
        self.raw_data_path = os.path.join(bufferPath, "{}_raw_data.pkl".format(os.path.basename(os.path.normpath(rawDataPath))))
        self.label_predict_iou_path = os.path.join(bufferPath, "{}_predict_label_iou.pkl".format(os.path.basename(os.path.normpath(rawDataPath))))
        
        #read raw data
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
        
        ## init size
        self.label_size = self.raw_labels[:,3]*self.raw_labels[:,4]
        self.predict_size = self.raw_predicts[:,4]*self.raw_predicts[:,5]
        ## init aspect ratio
        self.label_aspect_ratio = self.raw_labels[:,3]/self.raw_labels[:,4]
        self.predict_aspect_ratio = self.raw_predicts[:,4]/self.raw_predicts[:,5]
        ## TODO init direction
        
    def getMetaData(self):
        return {
            "hierarchy": self.hierarchy,
            "names": self.names
        }
            
    def compute_label_predict_pair(self):
        def compute_per_image(detections, labels):
            # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
            iou = box_iou(detections[:, 1:5], labels[:, 1:])

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
            "label_aspect_ratio": [0,1],
            "predict_aspect_ratio": [0,1],
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
            # size
            label_selected = np.logical_and(self.label_size>=query["label_size"][0], self.label_size<=query["label_size"][1])
            predict_selected = np.logical_and(self.predict_size>=query["predict_size"][0], self.predict_size<=query["predict_size"][1])
            # aspect ratio
            label_selected = np.logical_and(label_selected, np.logical_and(self.label_aspect_ratio>=query["label_aspect_ratio"][0], self.label_aspect_ratio<=query["label_aspect_ratio"][1]))
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
        
        return filtered, unmatch_predict, unmatch_label
    
    def getStatisticsMatrix(self, matrix, query):
        """
            matrix: a 3-d list consists of lists of indexes from predict_label_pairs 
        """
        statistics_mode = 'count'
        if query is not None and "return" in query:
            statistics_mode = query['return']
        stat_matrix = np.zeros((len(matrix), len(matrix[0])), dtype=np.float64)
        function_map = {
            'count': lambda x: len(x),
            'avg_label_size': lambda x: 0 if self.predict_label_pairs[x[0],1]==-1 else self.label_size[self.predict_label_pairs[x, 1]].mean(),
            'avg_predict_size': lambda x: 0 if self.predict_label_pairs[x[0],0]==-1 else self.predict_size[self.predict_label_pairs[x, 0]].mean(),
            'avg_iou': lambda x: self.predict_label_ious[x].mean(),
            'avg_acc': lambda x: 0 if np.any(self.predict_label_pairs[x[0],:]==-1) else 
                            (self.raw_predicts[self.predict_label_pairs[x,0], 0]==self.raw_labels[self.predict_label_pairs[x,1], 0]).mean(),
            'avg_label_aspect_ratio': lambda x: 0 if self.predict_label_pairs[x[0],1]==-1 else self.label_aspect_ratio[self.predict_label_pairs[x, 1]].mean(),
            'avg_predict_aspect_ratio': lambda x: 0 if self.predict_label_pairs[x[0],0]==-1 else self.predict_aspect_ratio[self.predict_label_pairs[x, 0]].mean(),
        }
        if statistics_mode not in function_map:
            raise NotImplementedError()
        map_func = function_map[statistics_mode]
        for i in range(stat_matrix.shape[0]):
            for j in range(stat_matrix.shape[1]):
                if len(matrix[i][j]) == 0:
                    continue
                stat_matrix[i, j] = map_func(matrix[i][j])
        return stat_matrix

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
        return self.getStatisticsMatrix(confusion, query).tolist()
    
    def getSizeMatrix(self, query = None):
        """A size matrix divided by iou with Fisher algorithm. 
        Samples are sorted by their pred size, and then calculate the split by Fisher algorithm.

        Args:
            querys (dict): {label/predict size:[a, b] (0<=a<=b<=1), label/predict aspect_ratio:[a, b] (0<=a<=b), direction: [0,..,8],
                            label/predict: np.arange(80), split: int}
        """
        K = 10
        filtered , unmatch_predict, unmatch_label = self.filterSamples(query)
        if query is not None and 'split' in query:
            K = query["split"]
        size_matrix = [[[] for _ in range(K+1)] for _ in range(K+1)]
        pred_size_argsort = np.argsort(self.predict_size)
        split_pos = get_size_split_pos(self.predict_label_ious[pred_size_argsort], K)
        split_size = self.predict_size[pred_size_argsort][split_pos]
        # print(split_size)
        label_split_rec, pred_split_rec = [], []

        for i in range(K):
            label_split = np.arange(len(self.label_size))
            pred_split = np.arange(len(self.predict_size))
            if i > 0:
                label_split = label_split[self.label_size[label_split] >= split_size[i-1]]
                pred_split = pred_split[self.predict_size[pred_split] >= split_size[i-1]]
            if i < K-1:
                label_split = label_split[self.label_size[label_split] < split_size[i]]
                pred_split = pred_split[self.predict_size[pred_split] < split_size[i]]
            label_split_rec.append(label_split)
            pred_split_rec.append(pred_split)
        for i in range(K):
            for j in range(K):
                size_matrix[i][j] = filtered[np.logical_and(np.isin(self.predict_label_pairs[filtered][:,1], label_split_rec[i]), 
                                                            np.isin(self.predict_label_pairs[filtered][:,0], pred_split_rec[j]))]
        for i in range(K):
            size_matrix[i][K] = unmatch_label[np.isin(self.predict_label_pairs[unmatch_label][:, 1], label_split_rec[i])]
            size_matrix[K][i] = unmatch_predict[np.isin(self.predict_label_pairs[unmatch_predict][:, 0], pred_split_rec[i])]
        return {
            'partitions': [0] + split_size.tolist() + [1],
            'matrix': self.getStatisticsMatrix(size_matrix, query).tolist()
        }
        
        
        
def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
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
    dataCtrler.process("/data/zhaowei/ConfusionMatrix//datasets/coco/", "/data/zhaowei/ConfusionMatrix/backend/buffer/")
    for rt_type in ['count','avg_label_size','avg_predict_size','avg_iou','avg_acc','avg_label_aspect_ratio','avg_predict_aspect_ratio']:
        matrix = dataCtrler.getSizeMatrix({
            "label_size": [0,1],
            "predict_size": [0,1],
            "label_aspect_ratio": [0,1],
            "predict_aspect_ratio": [0,1],
            "direction": [0,1,2,3,4,5,6,7,8],
            "label": np.arange(80),
            "predict": np.arange(80),
            "return": rt_type,
            "split": 10
        })
        print(rt_type, matrix)