#利用朴素贝叶斯进行20类新闻文本的处理

from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')
print(len(news.data))
