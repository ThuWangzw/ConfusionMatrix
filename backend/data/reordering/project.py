from distutils.log import error
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding

class Project:

    methods = {
        'pca': PCA, 
        'kpca': KernelPCA,
        'mds': MDS,
        'tsne': TSNE,
        'isomap': Isomap,
        'lle': LocallyLinearEmbedding
    }

    def __init__(self, matrix) -> None:
        self.matrix = np.array(matrix) if type(matrix)==list else matrix
    
    def get_proj(self, method='pca'):
        if method.lower() not in self.methods:
            raise NotImplementedError
        proj = self.methods[method.lower()](n_components=2)
        return proj.fit_transform(self.matrix)

    def get_proj_dm(self, DM, method="tsne"):
        if method.lower() not in self.methods:
            raise NotImplementedError
        proj = self.methods[method.lower()](n_components=2, metric="precomputed", random_state=100)
        return proj.fit_transform(DM)

    
        