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

```jupyter
import pandas as pd
import numpy as np
df = pd.read_csv('name.csv.gz')
```

```jupyter
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
