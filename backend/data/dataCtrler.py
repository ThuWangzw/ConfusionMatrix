import os
import json
import data.reorder as reorder

class DataCtrler(object):

    def __init__(self):
        super().__init__()
        self.statistic = {}
        self.labels = None
        self.preds = None
        self.features = None
        self.trainImages = None

    def processStatisticData(self, data):     
        return data

    def process(self, statisticData, predictData = None, trainImages = None, reordered=True):
        """process raw data
        """        
        self.statistic = self.processStatisticData(statisticData)

        if reordered:
            hierarchy_path = os.path.join('backend', 'data', 'hierarchy.json')
            if not os.path.exists(hierarchy_path):
                ordered_hierarchy = reorder.getOrderedHierarchy(self.statistic["confusion"])
                with open(hierarchy_path, 'w') as f:
                    json.dump(ordered_hierarchy, f)
            with open(hierarchy_path) as fr:
                ordered_hierarchy = json.load(fr)
            self.statistic["confusion"]['hierarchy'] = ordered_hierarchy

        if predictData is not None:
            self.labels = predictData["labels"].astype(int)
            self.preds = predictData["preds"].astype(int)
            self.features = predictData["features"]
            
        self.trainImages = trainImages
        
    def getConfusionMatrix(self):
        """ confusion matrix
        """        
        return self.statistic["confusion"]

    def getImagesInConsuionMatrixCell(self, labels: list, preds: list) -> list:
        """return images in a cell of confusionmatrix

        Args:
            labels (list): true labels of corresponding cell
            preds (list): predicted labels of corresponding cell

        Returns:
            list: images' id
        """ 
        # convert list of label names to dict
        labelNames = self.statistic['confusion']['names']
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
        if self.labels is not None and self.preds is not None:
            n = len(self.labels)
            for i in range(n):
                if self.labels[i] in labelSet and self.preds[i] in predSet:
                    imageids.append(i)
                    
        # limit length of images
        return imageids[:50]
    
    def getImage(self, imageID: int) -> list:
        """ get image by id

        Args:
            imageID (int): image id

        Returns:
            list: image
        """        
        if self.trainImages is not None:
            image = self.trainImages[imageID]
            return image.tolist()
        else:
            return []
    

dataCtrler = DataCtrler()