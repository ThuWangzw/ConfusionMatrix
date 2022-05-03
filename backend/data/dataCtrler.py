import os
import json
import numpy as np
# import data.reorder as reorder
import pickle
import torch
from sklearn.metrics import confusion_matrix

class DataCtrler(object):

    def __init__(self):
        super().__init__()
        self.iou_threshold = 0.45
        self.conf_threshold = 0.25
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
        self.raw_data_path = os.path.join(bufferPath, "{}_raw_data.pkl".format(os.path.basename(os.path.normpath(rawDataPath))))
        
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
                
            
        # compute (predict,label) pair
        self.compute_label_predict_pair()
        
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
            detections = detections[detections[:, 5] > self.conf_threshold]
            gt_classes = labels[:, 0].astype(np.int32)
            detection_classes = detections[:, 0].astype(np.int32)
            iou = box_iou(labels[:, 1:], detections[:, 1:5])

            x = np.where(iou > self.iou_threshold)
            if x[0].shape[0]:
                matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            else:
                matches = np.zeros((0, 3))
            final_match = -1*np.ones((len(labels),2), dtype=np.int32)
            ious = np.zeros(len(labels))
            final_match[:,0] = np.arange(len(labels), dtype=np.int32)
            for match in matches:
                final_match[int(match[0])] = match[:2].astype(np.int32)
                ious[int(match[0])] = match[2]
            return final_match, ious
        self.label_predict_pairs = -1*np.ones((len(self.raw_labels), 2), dtype=np.int32)
        self.label_predict_ious = np.zeros(len(self.raw_labels))
        for imageidx in range(len(self.image2index)):
            matches, ious = compute_per_image(self.raw_predicts[self.imageid2raw_predict[imageidx][0]:self.imageid2raw_predict[imageidx][1]],
                                        self.raw_labels[self.imageid2raw_label[imageidx][0]:self.imageid2raw_label[imageidx][1]])
            negaWeights = np.where(matches[:,1]==-1)[0]
            if len(matches)>0:
                matches[:,1]+=self.imageid2raw_predict[imageidx][0]
                matches[:,0]+=self.imageid2raw_label[imageidx][0]
                matches[negaWeights,1]=-1
                self.label_predict_pairs[self.imageid2raw_label[imageidx][0]:self.imageid2raw_label[imageidx][1]] = matches
                self.label_predict_ious[self.imageid2raw_label[imageidx][0]:self.imageid2raw_label[imageidx][1]] = ious
        return self.label_predict_pairs, self.label_predict_ious
            
    def getConfusionMatrix(self, query = None):
        """filtered confusion matrix

        Args:
            querys (dict): {label/predict size:[a, b], label/predict aspect_ratio:[a, b], direction: [0,..,8]}
        """        
        # remove fn
        filtered = self.label_predict_pairs[np.where(self.label_predict_pairs[:,1]>-1)[0]]
        fn = self.label_predict_pairs[np.where(self.label_predict_pairs[:,1]==-1)[0]]
        
        # confusion
        y_true = np.concatenate((self.raw_labels[filtered[:,0],0].astype(np.int32), self.raw_labels[fn[:,0],0].astype(np.int32)))
        y_predict = np.concatenate((self.raw_predicts[filtered[:,1],0].astype(np.int32), (len(self.classID2Idx)-1)*np.ones(len(fn), dtype=np.int32)))
        class_labels = np.zeros(len(self.classID2Idx))
        for id,idx in self.classID2Idx.items():
            class_labels[idx]=id
        confusion = confusion_matrix(y_true,y_predict,labels = np.arange(len(self.classID2Idx)))
        return confusion.tolist()
        
        
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
    matrix = dataCtrler.getConfusionMatrix({})
    print(matrix)