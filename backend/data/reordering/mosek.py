from mosek.fusion import *
import numpy as np
import sys
dim = 4
bw = dim
dis_mat = np.random.uniform(0, 100, [dim, dim])
dis_mat = dis_mat + dis_mat.T

mea_mat = np.zeros((dim, dim))
for i in range(dim):
    for j in range(dim):
        if abs(i-j)<=bw:
            mea_mat[i][j] = bw+1-abs(i-j)
        else:
            mea_mat[i][j] = 0

arr = np.array([i for i in range(dim)]).reshape([1, dim])

lca = np.array([
    [0, 2, 1, 0],
    [2, 0, 3, 1],
    [1, 3, 0, 4],
    [0, 1 ,4, 0]
])

M = Model("example model")


P = M.variable('P', [dim, dim], Domain.integral(Domain.inRange(0.,1.)))

t = M.variable('ans', 1, Domain.unbounded())

D = M.parameter('D', [dim, dim])
D.setValue(dis_mat)

A = M.parameter('A', [dim, dim])
A.setValue(mea_mat)

for a in range(dim):
    for b in range(dim):

        M.constraint(Expr.sub(Expr.sub(Expr.mulElm(P.slice([a, 0], [a+1, dim]), arr), Expr.mulElm(P.slice([b, 0], [b+1, dim]), arr)), lca[a][b]-1), Domain.lessThan(0.0))
        M.constraint(Expr.add(Expr.sub(Expr.mulElm(P.slice([a, 0], [a+1, dim]), arr), Expr.mulElm(P.slice([b, 0], [b+1, dim]), arr)), lca[a][b]-1), Domain.greaterThan(0.0))

M.constraint(t-Expr.sum(Expr.mulElm(Expr.mul(Expr.mul(P, Expr.mul(dis_mat, P.transpose()))), mea_mat)), Domain.equalsTo(0.))

M.objective(ObjectiveSense.Minimize, t)
# Expr.sum(Expr.eleMul(Expr.mul(Expr.mul(P, dis_mat), P.transpose()), mea_mat))
M.solve()

# n=10
# m=10
# M = Model('TV')

# u = M.variable("u", [n+1,m+1], Domain.inRange(0.,1.0) )
# t = M.variable("t", [n,m], Domain.unbounded() )

# # In this example we define sigma and the input image f as parameters
# # to demonstrate how to solve the same model with many data variants.
# # Of course they could simply be passed as ordinary arrays if that is not needed.
# sigma = M.parameter("sigma")
# f = M.parameter("f", [n,m])

# ucore = u.slice( [0,0], [n,m] ) 

# deltax = Expr.sub( u.slice( [1,0], [n+1,m] ), ucore)
# deltay = Expr.sub( u.slice( [0,1], [n,m+1] ), ucore)