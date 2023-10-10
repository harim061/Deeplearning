from sklearn import datasets


news=datasets.fetch_20newsgroups(subset='train')
print(news.DESCR)