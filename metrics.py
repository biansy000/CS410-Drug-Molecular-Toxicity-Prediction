import torch
import numpy as np

class EvaMetric:
    def __init__(self, pos_weight=7.0):
        self.total_num = 0
        self.truth_pos = 0.
        self.false_pos = 0.
        self.truth_neg = 0.
        self.false_neg = 0.
        self.weight = pos_weight
        #self.total_size = 0.
    
    def update(self, pred, gt, pos=1.0, neg=0.0):
        pred = pred.reshape(-1).detach().cpu().numpy()
        gt = gt.reshape(-1).detach().cpu().numpy()
        number = gt.shape[0]
        self.total_num += number
        #print(number, pred.shape, gt.shape)

        thres = (pos + neg)/2

        truth_pos = (gt > thres) & (pred > thres)
        false_pos = (gt <= thres) & (pred > thres)
        truth_neg = (gt <= thres) & (pred <= thres)
        false_neg = (gt > thres) & (pred <= thres)

        self.truth_pos += np.sum(truth_pos * 1.0)
        self.false_pos += np.sum(false_pos * 1.0)
        self.truth_neg += np.sum(truth_neg * 1.0)
        self.false_neg += np.sum(false_neg * 1.0)
        assert float(self.total_num) == self.truth_pos + self.false_pos + self.truth_neg + self.false_neg

    def accuracy(self):
        return (self.truth_pos + self.truth_neg) / self.total_num
         
    def precision(self):
        return self.truth_pos / (self.truth_pos + self.false_pos + 1e-8)
    
    def recall(self):
        return self.truth_pos / (self.truth_pos + self.false_neg + 1e-8)
    
    def weighted_accuracy(self):
        total = self.truth_pos * self.weight + self.false_pos + self.truth_neg + self.false_neg * self.weight
        #print('total', total)
        return (self.truth_pos * self.weight + self.truth_neg) / total
    
    def flush(self):
        self.total_num = 0
        self.truth_pos = 0.
        self.false_pos = 0.
        self.truth_neg = 0.
        self.false_neg = 0.


class AUCMetric:
    def __init__(self):
        self.gt_label = []
        self.pred = []
    
    def update(self, pred, gt):
        pred = pred.reshape(-1).detach().cpu().numpy()
        gt = gt.reshape(-1).detach().cpu().numpy()

        self.gt_label = self.gt_label + list(gt)
        self.pred = self.pred + list(pred)
    
    def AUC(self):
        data = [(item1, item2) for item1, item2 in zip(self.pred, self.gt_label)]
        def takeFst(a):
            return a[0]

        data.sort(key=takeFst, reverse = True)

        true_num = np.sum(np.array(self.gt_label) > 0.5)
        false_num = np.sum(np.array(self.gt_label) <= 0.5)

        true_pos = 0
        false_pos = 0
        space = 0
        for item in data:
            if item[1] == 1:
                true_pos += 1
            else:
                false_pos += 1
                add_space = 1 / false_num * true_pos / true_num
                space += add_space

        assert false_pos == false_num
        return space
            
