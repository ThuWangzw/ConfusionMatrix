import numpy as np

def I(x, y):
    return 1 if x>y else 0
def sign(x):
    if x>0: return 1
    if x==0: return 0
    return -1

class Measure:
    def __init__(self, mat) -> None:
        if type(mat)==list: mat = np.array(mat)
        self.mat = mat
        self.n = len(self.mat)

    def get_gradient_criterion(self):
        event = 0
        devia = 0
        grad  = 0
        w_grad = 0
        for i in range(self.n):
            for k in range(i+1,self.n):
                for j in range(k+1, self.n):
                    event += I(self.mat[i][k], self.mat[i][j]) + I(self.mat[k][j], self.mat[i][j])
                    devia += I(self.mat[i][k], self.mat[i][j]) * (self.mat[i][j]-self.mat[i][k]) + I(self.mat[k][j], self.mat[i][j]) * (self.mat[i][j]-self.mat[k][j])
                    grad += -sign(self.mat[i][j]-self.mat[i][k]) - sign(self.mat[i][j]-self.mat[k][j])
                    w_grad += -sign(self.mat[i][j]-self.mat[i][k])*abs(self.mat[i][j]-self.mat[i][k]) - sign(self.mat[i][j]-self.mat[k][j])*abs(self.mat[i][j]-self.mat[k][j])

        return {
            'AR_events': event,
            'AR_deviation': devia,
            'Gradient': grad,
            'weighted_grad': w_grad
        }

    def get_rank_criterion(self):
        lsqu = 0
        interia = 0
        two_sum = 0
        lser = 0
        for i in range(self.n):
            for j in range(self.n):  
                lsqu += (self.mat[i][j]-abs(i-j))**2
                interia += self.mat[i][j]*((i-j)**2)
                two_sum += (i-j)**2 / (1+self.mat[i][j])
                lser += self.mat[i][j]*abs(i-j)
        return { 
                'least_square': lsqu, 
                'interia': interia, 
                '2_sum': two_sum,
                'least_seriation': lser
        }

    def get_path_criterion(self):
        h_path = 0
        lazy_path = 0
        for i in range(self.n):
            if i+1<self.n:
                h_path += self.mat[i][i+1]
                lazy_path += self.mat[i][i+1] * (self.n-i-1)
        return {
            'path': h_path,
            'lazy_path': lazy_path
        }

    def get_neighbor_criterion(self):
        n_nei = self.n
        result = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                result += self.mat[i][j] / abs(i-j)
        return {
            'n_adj': result
        }

    def get_general_criterion(self):
        la = 0
        for i in range(self.n):
            for j in range(self.n):
                if self.mat[i][j] != 0:
                    la += abs(i-j)
        return {
            'Linear Arrangement': la
        }

    def get_BAR(self, bw=-1):
        if bw==-1: bw = self.n - 1
        mea_mat = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if abs(i-j)<=bw:
                    mea_mat[i][j] = bw+1-abs(i-j)
                else:
                    mea_mat[i][j] = 0
        bar = 0
        for i in range(self.n):
            for j in range(self.n):
                if i!=j:
                    bar += self.mat[i][j] * mea_mat[i][j]
        return {
            'BAR': bar
        }


    def get_measure(self, bw=-1):
        # gradient_criterion = self.get_gradient_criterion()
        # rank_criterion = self.get_rank_criterion()
        # path_criterion = self.get_path_criterion()
        # nei_criterion = self.get_neighbor_criterion()
        bar_criterion = self.get_BAR(bw)
        return bar_criterion
        return nei_criterion
        return [gradient_criterion, rank_criterion, path_criterion]