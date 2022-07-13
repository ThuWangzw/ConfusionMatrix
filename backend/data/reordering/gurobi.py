import gurobipy as gp
from gurobipy import GRB
import numpy as np

dim = 10
bw = dim
dis_mat = np.array([
    [0, 2, 1, 0],
    [2, 0, 3, 1],
    [1, 3, 0, 4],
    [0, 1 ,4, 0]
])

dis_mat = np.random.uniform(0, 1000, [dim, dim])
dis_mat = dis_mat + dis_mat.T

# lca_mat = np.array([
#     [4, 4, 4, 4],
#     [4, 4, 4, 4],
#     [4, 4, 4, 4],
#     [4, 4, 4, 4]
# ])
lca_mat = np.zeros([dim, dim])
for i in range(dim):
    for j in range(dim):
        lca_mat[i][j] = dim
# for i in range(dim//4):
#     for j in range(dim//4):
#         lca_mat[i][j] = dim//4
#         lca_mat[dim//4+i][dim//4+j] = dim//4
#         lca_mat[(dim//4)*2+i][(dim//4)*2+j] = dim//4
#         lca_mat[(dim//4)*3+i][(dim//4)*3+j] = dim//4
mea_mat = np.zeros((dim, dim))
for i in range(dim):
    for j in range(dim):
        if abs(i-j)<=bw:
            mea_mat[i][j] = bw+1-abs(i-j)
        else:
            mea_mat[i][j] = 0

m = gp.Model('QAP')

P = m.addMVar((dim, dim), vtype=GRB.BINARY)

for a in range(dim):
    for b in range(dim):
        m.addConstr( (gp.quicksum(P[a][j]*j for j in range(dim)) - gp.quicksum(P[b][j]*j for j in range(dim)) - lca_mat[a][b] <= -1))
        m.addConstr( (gp.quicksum(P[a][j]*j for j in range(dim)) - gp.quicksum(P[b][j]*j for j in range(dim)) + lca_mat[a][b] >= 1))

m.addConstrs((gp.quicksum(P[i])==1 for i in range(dim)))
m.addConstrs((gp.quicksum(P[:,j])==1 for j in range(dim)))

m.setObjective(sum((P[i] @ dis_mat @ P[j]) * mea_mat[i][j] for i in range(dim) for j in range(dim)), GRB.MINIMIZE)

m.optimize()
print(P.x)

def get_lca(tree, lca_mat):
    if len(tree.children)==0:
        lca_mat[tree.id][tree.id] = 1
        return 
    num_leaves = len(tree.leaves)
    for i in range(len(tree.children)):
        for j in range(i+1, len(tree.children)):
            ti = tree.children[i]
            tj = tree.children[j]
            for li in ti.leaves:
                for lj in tj.leaves:
                    lca_mat[li.id][lj.id] = num_leaves
                    lca_mat[lj.id][li.id] = num_leaves
    for c in tree.children:
        get_lca(c, lca_mat)

def get_order(tree, DM):

    dim = len(tree.leaves)
    bw = dim

    dis_mat = DM


    mea_mat = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if abs(i-j)<=bw:
                mea_mat[i][j] = bw+1-abs(i-j)
            else:
                mea_mat[i][j] = 0

    lca_mat = np.zeros((dim, dim))
    get_lca(tree, lca_mat)



    # MIP  model formulation

    m = gp.Model('QAP')

    P = m.addMVar((dim, dim), vtype=GRB.BINARY)

    for a in range(dim):
        for b in range(dim):
            m.addConstr( (gp.quicksum(P[a][j]*j for j in range(dim)) - gp.quicksum(P[b][j]*j for j in range(dim)) - lca_mat[a][b] <= -1))
            m.addConstr( (gp.quicksum(P[a][j]*j for j in range(dim)) - gp.quicksum(P[b][j]*j for j in range(dim)) + lca_mat[a][b] >= 1))

    m.addConstrs((gp.quicksum(P[i])==1 for i in range(dim)))
    m.addConstrs((gp.quicksum(P[:,j])==1 for j in range(dim)))

    m.setObjective(sum((P[i] @ dis_mat @ P[j]) * mea_mat[i][j] for i in range(dim) for j in range(dim)), GRB.MINIMIZE)

    m.optimize()
    print(P.x)

# M = Model("example model")


# P = M.variable('P', [dim, dim], Domain.integral(Domain.inRange(0.,1.)))

# t = M.variable('ans', 1, Domain.unbounded())

# D = M.parameter('D', [dim, dim])
# D.setValue(dis_mat)

# A = M.parameter('A', [dim, dim])
# A.setValue(mea_mat)

# for a in range(dim):
#     for b in range(dim):

#         M.constraint(Expr.sub(Expr.sub(Expr.mulElm(P.slice([a, 0], [a+1, dim]), arr), Expr.mulElm(P.slice([b, 0], [b+1, dim]), arr)), lca_mat[a][b]-1), Domain.lessThan(0.0))
#         M.constraint(Expr.add(Expr.sub(Expr.mulElm(P.slice([a, 0], [a+1, dim]), arr), Expr.mulElm(P.slice([b, 0], [b+1, dim]), arr)), lca_mat[a][b]-1), Domain.greaterThan(0.0))

# M.constraint(t-Expr.sum(Expr.mulElm(Expr.mul(Expr.mul(P, Expr.mul(dis_mat, P.transpose()))), mea_mat)), Domain.equalsTo(0.))

# M.objective(ObjectiveSense.Minimize, t)
# # Expr.sum(Expr.eleMul(Expr.mul(Expr.mul(P, dis_mat), P.transpose()), mea_mat))
# M.solve()

