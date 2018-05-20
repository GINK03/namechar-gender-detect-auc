import pandas as pd

df = pd.read_csv('ys.csv')

for th in [ 0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.8, 1]:
  tp, tn, fp, fn = 0, 0, 0, 0
  for o in df.to_dict('record'):
    #print(th, o)
    real = o['gender']
    pred = 1 if o['predict'] >= th else 0

    if real == 0 and pred == 1:
      fp += 1
    if real == 1 and pred == 0:
      fn += 1
    
    if real == 1 and pred == 1:
      tp += 1
    if real == 0 and pred == 0:
      tn += 1
  print('ratio_tp', tp/(tp+fn), 'ratio_fp', fp/(fp+tn))
