import trees
import plotTrees
file = 'D:\python_deep learning\Machine-Learning\Code-ML-ShiZhan\代码实现\决策树/lenses.txt'
fr = open(file)
lenses = [inst.strip().split('\t') for inst in fr.readlines() ]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = trees.createTree(lenses,lensesLabels)
print(lensesTree)
plotTrees.createPlot(lensesTree)