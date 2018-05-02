from sklearn import linear_model
reg = linear_model.LinearRegression()
a = [[0,0],
     [1,1],
     [2,2]
     ]

b = [0,1,2]

reg.fit(a,b)
print(reg.coef_)
