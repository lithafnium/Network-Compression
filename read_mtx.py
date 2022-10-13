from scipy.io import mmread

a = mmread("socfb-Harvard1.mtx")
print(a.toarray())
