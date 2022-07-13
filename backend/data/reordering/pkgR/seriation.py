# import rpy2
# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr
import copy
import matplotlib.pyplot as plt    # 绘图库
import numpy as np
import datetime
from sklearn.linear_model import ridge_regression

from sklearn.model_selection import RandomizedSearchCV
from tomlkit import date

# https://www.rdocumentation.org/packages/seriation/versions/1.3.5/topics/seriate
# https://www.rdocumentation.org/packages/seriation/versions/1.3.5/topics/criterion


r_seriation_methods_dist = ['ARSA', 'TSP', 'R2E', 'MDS', 'HC', 'HC_single', 'HC_complete', 'HC_average', 'HC_ward', 'GW', 'OLO', 'VAT', 'SA', 'GA', 'QAP_LS', 'QAP_2SUM', 'QAP_BAR', 'QAP_Inertia', 'SPIN_STS', 'SPIN_NH', 'Spectral', 'BBURCG', 'BBWRCG']
r_seriation_methods_mat = ['BEA', 'BEA_TSP', 'Heatmap', 'PCA', 'PCA_angle']
r_biclust_methods = ['BCPlaid', 'BCBimax', 'BCQuest', 'BCSpectral']
r_criterion = ["Gradient_raw", "Gradient_weighted", "AR_events", "AR_deviations", "RGAR", "BAR", "Path_length", "Lazy_path_length", "Inertia", "Least_squares", "LS", "2SUM", "ME", "Moore_stress", "Neumann_stress", "Cor_R"]
r_null = robjects.NULL

# file_path = './data/reordering/pkgR/reordering.R'
file_path = './backend/data/reordering/pkgR/reordering.R'

def plot_confusion_matrix(cm, labels_name, title):
    cm = cm / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.clf()
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_class = np.array(range(len(labels_name)))#获取标签的间隔数    
    plt.xticks(num_class, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_class, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
    plt.savefig(title + "_" + datetime.datetime.now().strftime('%R-%S')+'.png')

def get_r_mat(matrix):
    """
    matrix 2d list
    """
    vec = []
    n_row = len(matrix)
    for row in matrix: vec.extend(row)
    r_vec = robjects.FloatVector(vec)
    r_mat = robjects.r['matrix'](r_vec, nrow = n_row, byrow=True)
    return r_mat

def capture_r_output():
    """
    Will cause all the output that normally goes to the R console,
    to end up instead in a python list.
    """
    # Import module #
    import rpy2.rinterface_lib.callbacks
    # Record output #
    stdout = []
    stderr = []
    # Dummy functions #
    def add_to_stdout(line): stdout.append(line)
    def add_to_stderr(line): stderr.append(line)
    # Keep the old functions #
    stdout_orig = rpy2.rinterface_lib.callbacks.consolewrite_print
    stderr_orig = rpy2.rinterface_lib.callbacks.consolewrite_warnerror
    # Set the call backs #
    rpy2.rinterface_lib.callbacks.consolewrite_print     = add_to_stdout
    rpy2.rinterface_lib.callbacks.consolewrite_warnerror = add_to_stderr

def getSeriationOrder(matrix, method="OLO"):
    """"
    matrix: 2d list
    method: one of r_seriation_methods_dist/mat
    """
    capture_r_output()
    robjects.r.source(file_path)
    r_mat = get_r_mat(matrix)

    if method=='GA':
        perm = list(robjects.r.r_seriation_GA(r_mat, method))
    elif method=='DendSer':
        perm = list(robjects.r.r_seriation_DendSer(r_mat, method))
    elif method in r_seriation_methods_dist:
        perm = list(robjects.r.r_seriation_dist(r_mat, method))
    elif method in r_seriation_methods_mat:
        perm = list(robjects.r.r_seriation_mat(r_mat, method))
    else:
        raise NotImplementedError
    for i in range(len(perm)): perm[i] -= 1 
    
    ## debug
    # import numpy as np
    # matrix = np.array(matrix)
    # plot_confusion_matrix(matrix, list(range(len(matrix))), 'before')
    # print(getMetric(matrix))

    # matrix = matrix[perm][:,perm]
    # plot_confusion_matrix(matrix, list(range(len(matrix))), 'after')
    # print(getMetric(matrix))
    ##

    return perm

def getBiclustOrder(matrix, method="BCPlaid"):
    """"TBD
    """
    robjects.r.source(file_path)
    r_mat = get_r_mat(matrix)
    
    result = robjects.r.r_biclust(r_mat, method)
    result = list(result)
    n_cluster = len(result)
    row_indexes = []
    col_indexes = []
    for bicluster in result:
        row_indexes.append(list(bicluster[0]))
        col_indexes.append(list(bicluster[1]))
    perm = []
    for item in row_indexes: 
        for idx in item:
            if item not in perm: perm.append(idx)
    for i in range(1, len(matrix)+1):
        if i not in perm:
            perm.append(i)
    # print()

    for i in range(len(perm)): perm[i] -= 1 
    return perm

def getMetric(matrix, metric=r_null):
    """
    matrix: 2d list, distance matrix
    metric: r_null(all criterions) / one of r_criterion
    """
    robjects.r.source(file_path)
    
    r_mat = get_r_mat(matrix)

    if metric and metric not in r_criterion: raise NotImplementedError

    metrics = robjects.r.r_criterion(r_mat, metric)

    if metric == r_null:
        criterions = list(metrics.names)
        metrics = list(metrics)
        return dict([(k, v) for (k, v) in zip(criterions, metrics)])
    else:
        return list(metrics)[0]

def testMetric(matrix, metric=r_null, method=r_null):
    """Get metric of matrix reordered by method
    matrix: 2d list, distance matrix
    metric: r_null(all criterions) / one of r_criterion
    method: r_null(original matrix) / one of r_seriation_methods_dist/mat
    """
    robjects.r.source(file_path)
    
    r_mat = get_r_mat(matrix)

    if metric and metric not in r_criterion: raise NotImplementedError
    if method and method not in r_seriation_methods_dist: raise NotImplementedError

    metrics = robjects.r.r_test_criterion(r_mat, method=metric, order=method)

    if metric == r_null:
        criterions = list(metrics.names)
        metrics = list(metrics)
        return dict([(k, v) for (k, v) in zip(criterions, metrics)])
    else:
        return list(metrics)[0]


if __name__=="__main__":
    # getOrder(None)
    # getMetric([[1,3,2],[3,2,4],[2,4,3]], metric=r_null)
    # testMetric([[1,3,2, 4],[3,2,4, 5],[2,4,3, 7]], metric="AR_events", method='MDS')
    getBiclustOrder([[1,3,2, 4],[3,2,4, 5],[2,4,3, 7]])