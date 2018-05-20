# AUCを計算する

偽陽性率、真陽性率のバランスをとることで、間違って陽と判断してしまうリスクなどのバランスをとることで、誤って判別するリスクを最小化しながら、
正しく判別される率を担保することができます。  


## 問題設定
アメリカの各州の出生した人の性別と、名前と、その年の集計表のデータセットを利用します。  

[kaggleのopen data](https://www.kaggle.com/datagov/usa-names/data)で公開されており、誰でも参照利用することができます。  

LightGBMで名前のcharactor levelのベクトルと、州、年を特徴量として、分類してみましょう。  

この問題設定ではAUCを手計算することで、AUCが一体何なのかの理解を促進することを目的としています。  

## データをどうするか
LightGBMはカテゴリ変数を効率的に扱うことができ、ある特定の文字の組み合わせが、性別を説明する特徴を持つとき、勾配ブーストはその組み合わせ表現を獲得することができます。  

またCharactorLevel-CNNは位置がずれても組み合わせによる表現のパターンを獲得することができ、位置が変動するような。より長文では、CNNのほうがいいのですが、名前のようなものだと、勾配ブーストでも(それなりに)学習できます。  

<div align="center">
  <img width="750px" src="https://user-images.githubusercontent.com/4949982/40270042-b09aa9d6-5bc1-11e8-89e0-0e110d62bd93.png">
</div>
<div align="center"> 図1. こんなようなデータの持ち方にしました </div>

## 前処理

**Pandasでデータを読み取り、名前を文字粒度で分解して、インデックスを振り、カテゴリ変数にするために前処理を行います**  

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

**データセットをtrainとvalidationに分割してLightGBMのデータセットにします**  
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
## 学習
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
## 予想とAUCの計算

**validationを予想して、正解とのペアが記されたcsvを出力します**  
```python
lgb_clf = lgb.Booster(model_file='model')
ypr     = lgb_clf.predict(xva) 
ypr = pd.DataFrame(ypr)
ypr.columns = ['predict']
yva = pd.DataFrame(yva).reset_index()
yva = yva.drop(['index', 'level_0'], axis=1)

ys = pd.concat([yva, ypr], axis=1)
ys.to_csv('ys.csv', index=None)

ys.head()
```
<img width="174" alt="2018-05-20 0 28 25" src="https://user-images.githubusercontent.com/4949982/40270272-c4947c8e-5bc4-11e8-8760-ee725183d16f.png">

**validationでROC曲線を描く**  

２値分類では、真偽と判断されるしきい値を変化させながら、以下の表を、しきい値分、作成します  
```python
import pandas as pd
df = pd.read_csv('ys.csv')
for th in [ 0, 0.0001, 0.01, 0.1, 0.5, 0.8, 1]:
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
  print('ratio_tp', tp/(tp+tn), 'ratio_fp', fp/(fp+ft))
```
これを実行するとこのような結果を得られます。  
```conosle
$ python3 auc_calc.py

```

<div align="center">
  <img width="550px" src="https://user-images.githubusercontent.com/4949982/40270967-95131b2a-5bd1-11e8-82fa-30dbde675619.png">
</div>
しきい値を変化させると、a, b, c, dが変化するので、それをプロットすると、ROC曲線を描くことができます。

<div align="center">
  <img width="550px" src="https://user-images.githubusercontent.com/4949982/40271009-55b4ab78-5bd2-11e8-9659-289bc403cf21.png">
</div>
AUCの値とは、このカーブの下側の面積を指しており、1に近いほどよいモデルということができます。  

AUCの数値的な値自体は、２値分類においては、判別しきい値を変動させプロットさせるのですが、実際にはScikit-Learn等のソフトウェアで行うと、手っ取り早いです。

しきい値を連続的に変化させながら、判別の性能を見ると、しきい値が下がると右に全体のサンプル数が寄っていくことが確認できるかと思います。  
<div align="center">
  <img width="550px" src="https://user-images.githubusercontent.com/4949982/40276746-0d1e00a0-5c4d-11e8-8726-44d370a205d0.png">
</div>

**理想的な状態**   
<div align="center">
  <img width="550px" src="https://user-images.githubusercontent.com/4949982/40276864-7ea37578-5c4f-11e8-95ec-9aafda0d9636.png">
</div>
これが成立状態が理想的(現実的にはなかなかその通りになることはありません)  
