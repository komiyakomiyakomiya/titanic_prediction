# %%
# !pip install autopep8 japanize-matplotlib kaggle python-language-server pyls

# %%
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# import lightgbm as lgb
from IPython.display import display
import japanize_matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_profiling as pdp
import seaborn as sns
import time


# 日本語が豆腐になったらコレ
sns.set(font="IPAexGothic")
# データ読み込み
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')


# %%

# %%
# 概要
display(train.info())
print('#'*20)
display(test.info())


# %%
# 欠損値数
display(train.isnull().sum())
print('#'*20)
display(test.isnull().sum())


# %%
# 要約統計量（平均、標準偏差、最大値、最小値、最頻値など）
print(train.describe())
print('#'*20)
print(test.describe())
print('#'*20)
# percentilesで刻み位置を指定
print(train.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9]))
print('#'*20)
# include='O'でオブジェクト型の要素数、ユニーク数、最頻値、最頻値の出現回数を表示
print(train.describe(include='O'))


# %%
# join_train_test = pd.concat([train, test], axis=0, sort=False, join='inner')
# print(join_train_test.shape)
# display(join_train_test)
# join_train_test.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9])


# %%
# pdp.ProfileReport(train)


# %%　
# 可視化
# 生存
sns.countplot(x='Survived', data=train)
plt.title('死亡者と生存者の数')
plt.xticks([0, 1], ['死亡者', '生存者'])
plt.show()

display(train['Survived'].value_counts())
display(train['Survived'].count())  # 891
display(len(train['Survived']))  # 891
display(train['Survived'][train['Survived'] == 0].count())  # 549
display(train['Survived'][train['Survived'] == 1].count())  # 342
display(train['Survived'].value_counts() / train['Survived'].count())


# %%
# 性別
sns.countplot(x='Sex', hue='Survived', data=train)
plt.title('男女別の死亡者と生存者数')
plt.xticks([0, 1], ['男', '女'])
plt.legend(['死亡', '生存'])
plt.legend(bbox_to_anchor=(1, 1.2))  # original
plt.show()

display(pd.crosstab(train['Sex'], train['Survived']))
display(pd.crosstab(train['Sex'], train['Survived'], normalize='index'))


# %%
# チケットクラス
sns.countplot(x='Pclass', hue='Survived', data=train)
plt.title('チケットクラス別の死亡者と生存者数')
plt.xticks([0, 1, 2], ['上', '中', '下'])
plt.legend(['死亡', '生存'])
plt.show()

display(pd.crosstab(train['Pclass'], train['Survived']))
display(pd.crosstab(train['Pclass'], train['Survived'], normalize='index'))


# %%
# 年齢
sns.distplot(train['Age'].dropna(), kde=False, bins=30, label='全体')
sns.distplot(train[train['Survived'] == 0].Age.dropna(), kde=False, bins=30,
             label='死亡')
sns.distplot(train[train['Survived'] == 1]['Age'].dropna(),
             kde=False, bins=30, label='生存')
plt.title('乗船者の年齢の分布')
plt.legend()


# %%
train['CategoricalAge'] = pd.cut(train['Age'], 8)
display(train)
display(pd.crosstab(train['CategoricalAge'], train['Survived']))
display(pd.crosstab(train['CategoricalAge'],
                    train['Survived'], normalize='index'))
sns.countplot(x='CategoricalAge', hue='Survived', data=train)
plt.title('年代別生存者数')
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], ['0 - 10', '10 - 20', '20 - 30',
                                      '30 - 40', '40 - 50', '50 - 60', '60 - 70', '70 - 80'])
plt.legend(['死亡', '生存'])
plt.show()


# %%
# 同乗者
sns.countplot(x='SibSp', data=train)
plt.title('乗船している兄弟・配偶者の数')
plt.show()

display(pd.crosstab(train['SibSp'], train['Survived']))


# %%
train['SibSp_0_1_2over'] = [i if i <= 1 else 2 for i in train['SibSp']]
display(train['SibSp_0_1_2over'])
sns.countplot(x='SibSp_0_1_2over', hue='Survived', data=train)
plt.title('同乗している兄弟・配偶者の数と、生存・死亡')
plt.xticks([0, 1, 2], ['0人', '1人', '2人以上'])
plt.legend(['死亡', '生存'])
plt.show()

display(pd.crosstab(train['SibSp_0_1_2over'], train['Survived']))
display(pd.crosstab(train['SibSp_0_1_2over'],
                    train['Survived'], normalize='index'))


# %%
sns.countplot(x='Parch', data=train)
plt.title('同乗している両親・子供の数')


# %%
train['Parch_0_1_2_3over'] = [i if i <= 2 else 3 for i in train['Parch']]
sns.countplot(x='Parch_0_1_2_3over', hue='Survived', data=train)
plt.title('同乗している両親・子供の数')
plt.xticks([0, 1, 2, 3], ['0人', '1人', '2人', '3人以上'])
plt.legend(['死亡', '生存'])
plt.show()

display(pd.crosstab(train['Parch_0_1_2_3over'], train['Survived']))
display(pd.crosstab(train['Parch_0_1_2_3over'],
                    train['Survived'], normalize='index'))


# %%
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
display(train['FamilySize'])

sns.countplot(x='FamilySize', hue='Survived', data=train)
plt.title('家族の人数別の死亡者と生存者の数')
plt.legend(['死亡', '生存'])
plt.show()


# %%
train['IsAlone'] = [0 if i <= 1 else 1 for i in train['FamilySize']]
display(train['IsAlone'])
sns.countplot(x='IsAlone', hue='Survived', data=train)
plt.title('１人or２人以上で乗船別の死亡者と生存者の数')
plt.xticks([0, 1], ['1人', '2人以上'])
plt.legend(['死亡', '生存'])
plt.show()

display(pd.crosstab(train['IsAlone'], train['Survived']))
display(pd.crosstab(train['IsAlone'], train['Survived'], normalize='index'))


# %%
# 運賃
sns.distplot(train['Fare'], kde=False, hist=True)
plt.title('運賃の分布')
plt.show()


# %%
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
sns.countplot(x='CategoricalFare', hue='Survived', data=train)
plt.title('運賃別の分布別の死亡者と生存者の数')
plt.legend(['死亡', '生存'])
plt.show()


# %%
display(train[['CategoricalFare', 'Survived']])
display(train[['CategoricalFare', 'Survived']
              ].groupby(['CategoricalFare']).sum())
display(train[['CategoricalFare', 'Survived']
              ].groupby(['CategoricalFare']).mean())

display(pd.crosstab(train['CategoricalFare'], train['Survived']))
display(pd.crosstab(train['CategoricalFare'],
                    train['Survived'], normalize='index'))


# %%
# 名前
display(train['Name'][0:5])
# 敬称を抽出（重複削除）
display(set(train['Name'].str.extract('([A-Za-z]+)\.', expand=False)))
# 敬称別に人数をカウント　
display(train['Name'].str.extract(
    '([A-Za-z]+)\.', expand=False).value_counts())
# 敬称をのカテゴリ変数を追加　
train['Title'] = train['Name'].str.extract('([A-Za-z]+)\.', expand=False)
display(pd.crosstab(train['Title'], train['Survived']))
display(pd.crosstab(train['Title'], train['Survived'], normalize='index'))
# 敬称と年齢の関係を見る
display(train[['Title', 'Age']])
# 敬称の平均年齢を算出
# display(train[['Title', 'Age']].groupby('Title').mean())
display(train.groupby('Title').mean()['Age'])
# 敬称を数値に変換
# train['Title_num'] = [1 if i == 'Master' else 2 if i == 'Miss' else 3 if i == 'Mr' else 4 if i == 'Mrs' else 5 for i in train['Title']]


def title_to_num(title):
    if title == 'Master':
        return 1
    elif title == 'Miss':
        return 2
    elif title == 'Mr':
        return 3
    elif title == 'Mrs':
        return 4
    else:
        return 5


# 変換した変数列を追加
train['Title_num'] = [title_to_num(i) for i in train['Title']]
display(train['Title_num'])
display(pd.crosstab(train['Title_num'], train['Survived']))
display(pd.crosstab(train['Title_num'], train['Survived'], normalize='index'))

# testデータにもTitleカラムを追加
test['Title'] = test['Name'].str.extract('([A-Za-z]+)\.', expand=False)
test['Title_num'] = [title_to_num(i) for i in test['Title']]
display(test['Title_num'])

# %%
# SexとEmbarkedのOne-Hotエンコーディング
ohe_columns = ['Sex', 'Embarked']
train = pd.get_dummies(train, columns=ohe_columns)
test = pd.get_dummies(test, columns=ohe_columns)

display(train)
display('#'*20)
display(test)

# %%
# 不要カラムを削除
# inplaceをTrueにすると元のDataFrameが変更される
# この場合新しいDataFrameは返されず返り値はNone
train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

# %%
display(train)
display('#'*20)
display(test)

# %%
# X_trainをSurvivedカラム（正解データ）以外とする
X_train = train.drop(['Survived'], axis=1)
# y_trainをSurvivedカラム（正解データ）とする
y_train = train['Survived']


# %%
