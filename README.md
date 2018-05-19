# AUCを計算する

偽陽性率、真陽性率のバランスをとることで、間違って陽と判断してしまうリスクなどのバランスをとることで、誤って判別するリスクを最小化しながら、
正しく判別される率を担保することができます。  


## 問題設定
アメリカの各州の出生した人の性別と、名前と、その年の集計表を離島します。  

[kaggleのopen data](https://www.kaggle.com/datagov/usa-names/data)で誰でも参照利用することができます。  

LightGBMでcharactor levelと、州、年を特徴量として、分類してみます。  

AUCを手動で計算し、どういうことがよく見てみましょう  

## データの取扱　　
LightGBMはカテゴリ変数を効率的に扱うことができ、ある特定の文字の組み合わせが、性別的な特徴を持つとき、勾配ブーストはその組み合わせ表現を獲得することができます。  

またCharactorLevel-CNNは位置がずれても同じようなパターンを獲得することができ、より長文では、こちらのほうがいいのですが、名前のようなものだと、勾配ブーストでも(それなりに)学習できます。  

<div align="center">
  <img width="750px" src="https://user-images.githubusercontent.com/4949982/40270042-b09aa9d6-5bc1-11e8-89e0-0e110d62bd93.png">
</div>
<div align="center"> 図1. こんなようなデータの持ち方にしました </div>

## 前処理
Pandasでデータを読み取り、名前を文字粒度で分解して、インデックスを振り、カテゴリ変数にするために前処理を行います。  

```python
import pandas as pd
import numpy as np
df = pd.read_csv('name.csv.gz')
df.head()
```
<img width="288" alt="2018-05-20 0 14 18" src="https://user-images.githubusercontent.com/4949982/40270126-c8b1163a-5bc2-11e8-96fe-4916c3a6712d.png">

```python
# slicing name to map each index
char_index = {char:index+1 for index,char in enumerate('abcdefghijklmnopqrsutvwxyz')}
def slicer(i, name):
    try:
        char = name[i].lower()
        return char_index[char]
    except:
        return 0
for i in range(20):
    df[f'name_index_{i:04d}'] = df['name'].apply(lambda x:slicer(i, x)).astype(np.int16)
```

```python
state_index = { state:index for index, state in enumerate(set(df['state'].tolist())) }
df[f'state_index'] = df['state'].apply(lambda x:state_index[x]).astype(np.int16)
y =  df[ 'gender' ].apply(lambda x: 1.0 if x == "F" else 0.0)

predictors = [f'name_index_{i:04d}' for i in range(20)] + ['year', 'number', 'state_index']
categorical = [f'name_index_{i:04d}' for i in range(20)] + ['state_index']
from sklearn.model_selection import train_test_split
xtr, xva, ytr, yva = train_test_split(df.drop(['state', 'gender', 'name'], axis=1), y, test_size=0.10, random_state=23)

import lightgbm as lgb
lgtrain = lgb.Dataset(xtr, ytr.values,
                feature_name=predictors,
                categorical_feature = categorical)
lgvalid = lgb.Dataset(xva, yva.values,
                feature_name=predictors,
                categorical_feature = categorical)
```

```python
lgbm_params =  {
    'task'            : 'train',
    'boosting_type'   : 'gbdt',
    'objective'       : 'binary',
    'metric'          : 'auc',
    'max_depth'       : 15,
    'num_leaves'      : 33,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'learning_rate'   : 0.9,
    'verbose'         : 0
}  
lgb_clf = lgb.train(
    lgbm_params                     ,
    lgtrain                         ,
    num_boost_round=16000           ,
    valid_sets  = [lgtrain, lgvalid],
    valid_names = ['train','valid'] ,
    early_stopping_rounds=200       ,
    verbose_eval=200
)
lgb_clf.save_model('model')
```
