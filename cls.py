import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tqdm import trange, tqdm
from sklearn.model_selection import train_test_split


with open('./toy_sentences', 'r', encoding = 'utf-8') as f:
    all_sentences = f.readlines()


x = []
y = []
for line in tqdm(all_sentences):
    line = line.strip().replace(' ', '').replace('=', '+')
    ele  = line.split('+')
    a    = int(ele[0])
    b    = int(ele[1])
    c    = int(ele[2])
    x.append([a, b])
    y.append(c)
x = np.array(x)
y = np.array(y) - 1  # start from 0

x_tr, x_tst, y_tr, y_tst = train_test_split(x, y, test_size = 10000, shuffle = True, random_state = 42)

model = XGBClassifier(tree_method = 'gpu_hist', gpu_id = 0).fit(x_tr, y_tr)
py    = model.predict(x_tst)
acc   = 100 * (y_tst == py).mean()
print(f'ACC = {acc:.2f}%')



x = []
y = []
for line in tqdm(all_sentences):
    line = line.strip().replace(' ', '').replace('=', '+')
    ele  = line.split('+')
    a    = [int(v) for v in list(f'{int(ele[0]):07d}')]
    b    = [int(v) for v in list(f'{int(ele[1]):07d}')]
    c    = int(ele[2])
    x.append(a + b)
    y.append(c)
x = np.array(x)
y = np.array(y) - 1  # start from 0

x_tr, x_tst, y_tr, y_tst = train_test_split(x, y, test_size = 10000, shuffle = True, random_state = 42)

model = XGBClassifier(tree_method = 'gpu_hist', gpu_id = 0).fit(x_tr, y_tr)
py    = model.predict(x_tst)
acc   = 100 * (y_tst == py).mean()
print(f'ACC = {acc:.2f}%')
