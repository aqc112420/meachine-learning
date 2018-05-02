from sklearn.neighbors import BallTree
import numpy as np
np.random.seed(0)
X = np.random.random((10,3))
print(X)
tree = BallTree(X,leaf_size=2)
dist,ind = tree.query(X[[0]],k=4)
print(dist)
print(tree.query_radius(X[[0]],r=0.3,count_only=True))

