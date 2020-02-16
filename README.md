# こんなもん作りました

![zwmew-57a83.gif](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/287998/68d2f19b-c68c-8e81-0948-fb33381d0c7d.gif)

# なにやってるのか

1. モデルを作成
2. GCP にモデルをデプロイ
3. GCP 上で予測

※ タイタニックデータを利用しています。
※ 「年齢」「性別」だけを使い、生存できるかの２クラス分類モデルを作っています。

- GCP の AI-Platform にモデルをデプロイし予測します。

# GoogleColab

notebook にしましたので、よろしければご覧ください。
※ 2020/02/16 時点での情報及び動作確認となります。

https://colab.research.google.com/drive/1s6_o-nvMmHhAeBAOoRDgE0UNU6Fe3XrC

# 手順

notebook とほぼ同じ内容を記載しています。
Colab を開くのがめんどくさい方はこちらを御覧頂ければと思います。

<br>

## GCP アカウント登録

[【画像で説明】Google Cloud Platform (GCP)の無料トライアルでアカウント登録](https://qiita.com/komiya_____/items/14bd06d0866f182ae912)

<br>

## Google Cloud SDK のインストール

[Google Cloud SDK のインストール ~ 初期化](https://qiita.com/komiya_____/items/5af0dcc8639fad9fee29)

<br>

## SDK 認証

gcloud コマンドをつかって GCP をいじるため Google アカウントで認証します。

```bash
$ gcloud auth login
```

<br>

## プロジェクト作成

ID はカブり禁止です。

```bash
$ PROJECT_ID=anata-no-pj-id
$ PROJECT_NAME=anata-no-pj-name

$ gcloud projects create $PROJECT_ID \
--name $PROJECT_NAME
```

<br>

## 請求先アカウントの設定

設定しておかないとバケットにアクセスする際などで 403 エラーになります。

<img width="995" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/287998/d7f69423-b671-6816-e185-8b7ede2f7b5e.png">

以下のポップアップが出なければ、請求先アカウント設定済みなのでスキップして OK です。

<img width="876" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/287998/14f33fa4-7da2-2b02-206e-5feb5ece3944.png">

<img width="878" src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/287998/72de2c43-4306-0b51-819b-66f407979281.png">

<br>

## プロジェクトをアクティブに

コマンド操作の対象プロジェクトに設定する。

```bash
$ gcloud config set project $PROJECT_ID
```

<br>

## 確認

```bash
! gcloud config list

# [component_manager]
# disable_update_check = True
# [compute]
# gce_metadata_read_timeout_sec = 0
# [core]
# account = anata-no-address@gmail.com
# project = anata-no-pj-id
#
# Your active configuration is: [default]
```

- project = anata-no-project-id

となっていれば OK

<br>

## リージョン、ゾーン、インタープリタを設定

```bash
REGION=us-central1
ZONE=us-central1-a

$ gcloud config set compute/region $REGION
$ gcloud config set compute/zone $ZONE
$ gcloud config set ml_engine/local_python $(which python3)
```

AI Platform のオンライン予測が使えるリージョンは以下、、

- us-central1
- us-east1
- us-east4
- asia-northeast1
- europe-west1

インタープリタはローカルトレーニングの際に python ３系を使うように指定。

<br>

## 確認

```bash
$ gcloud config list

# [component_manager]
# disable_update_check = True
# [compute]
# gce_metadata_read_timeout_sec = 0
# region = us-central1
# zone = us-central1-a
# [core]
# account = anata-no-address@gmail.com
# project = anata-no-pj-id
# [ml_engine]
# local_python = /usr/bin/python3
#
# Your active configuration is: [default]
```

- region = us-central1
- zone = us-central1-a
- local_python = /usr/bin/python3

となっていれば OK

<br>

## トレーニング用のコード一式をクローン

https://github.com/komiyakomiyakomiya/titanic_prediction_on_gcp

```bash
$ git clone https://github.com/komiyakomiyakomiya/titanic_prediction_on_gcp.git
```

<br>

## model を保存するディレクトリを作成

```py:notebook
import os

os.makedirs('./titanic_prediction_on_gcp/working/models/', exist_ok=True)
```

<br>

## ローカルでトレーニング & モデルを保存

トレーニング済モデルが `./titanic_prediction_on_gcp/working/models/model.pkl` として保存されます。

```bash
$ gcloud ai-platform local train \
--package-path titanic_prediction_on_gcp/working/ \
--module-name working.predict_xgb
```

<br>

## バケットの作成

保存したモデルをアップロードするため、GCS にバケットを作ります。
※ 請求先アカウントを設定していないと 403 エラーになります。

```bash
BUCKET_NAME=anata-no-bkt-name

$ gsutil mb -l $REGION gs://$BUCKET_NAME
```

## 確認

```bash
$ gsutil ls -la

# gs://anata-no-bkt-name/
```

<br>

## 保存したモデルを GCS へアップロード

```bash
$ gsutil cp ./titanic_prediction_on_gcp/working/models/model.pkl gs://$BUCKET_NAME/models/model.pkl
```

## 確認

```bash
$ gsutil ls gs://$BUCKET_NAME/models/

# gs://anata-no-bkt-name/models/model.pkl
```

## API を有効化

AI-Platform API を使うためには以下の２つを有効化。

- ml.googleapis.com
- compute.googleapis.com

```bash
$ gcloud services enable ml.googleapis.com
$ gcloud services enable compute.googleapis.com
```

<br>

## 確認

```bash
$ gcloud services list --enabled

# NAME                              TITLE
# bigquery.googleapis.com           BigQuery API
# bigquerystorage.googleapis.com    BigQuery Storage API
# cloudapis.googleapis.com          Google Cloud APIs
# clouddebugger.googleapis.com      Stackdriver Debugger API
# cloudtrace.googleapis.com         Stackdriver Trace API
# compute.googleapis.com            Compute Engine API
# datastore.googleapis.com          Cloud Datastore API
# logging.googleapis.com            Stackdriver Logging API
# ml.googleapis.com                 AI Platform Training & Prediction API
# monitoring.googleapis.com         Stackdriver Monitoring API
# oslogin.googleapis.com            Cloud OS Login API
# servicemanagement.googleapis.com  Service Management API
# serviceusage.googleapis.com       Service Usage API
# sql-component.googleapis.com      Cloud SQL
# storage-api.googleapis.com        Google Cloud Storage JSON API
# storage-component.googleapis.com  Cloud Storage
```

- compute.googleapis.com Compute Engine API
- ml.googleapis.com AI Platform Training & Prediction API

があれば OK

<br>

## モデルリソース / バージョンリソースの作成

モデルリソースとバージョンリソースを作成し、アップロードした model.pkl と紐付けます。

モデルリソース

```bash
MODEL_NAME=model_xgb
MODEL_VERSION=v1

$ gcloud ai-platform models create $MODEL_NAME \
--regions $REGION
```

<br>

バージョンリソース

```bash
! gcloud ai-platform versions create $MODEL_VERSION \
--model $MODEL_NAME \
--origin gs://$BUCKET_NAME/models/ \
--runtime-version 1.14 \
--framework xgboost \
--python-version 3.5
```

<br>

## インプットデータの確認

あらかじめ用意していたデータで予測をしてみます。
まずは中身を確認。

```bash
! cat titanic_prediction_on_gcp/input/titanic/predict.json

# [36.0, 0] <- 36歳, 男性
```

[年齢, 性別]という形式で、性別は男性:0, 女性:1 とします。

<br>

## AI-Platform で予測

```bash
! gcloud ai-platform predict \
--model model_xgb \
--version $MODEL_VERSION \
--json-instances titanic_prediction_on_gcp/input/titanic/predict.json
```

[0.44441232085227966]　こんな予測値が返ってくれば OK。

<br>

## サービスアカウントの作成

次は python から AI-Platform にアクセスし、予測を取得します。
サービスアカウントキーが必要になるので、まずはサービスアカウントを作成。

```bash
SA_NAME=anata-no-sa-name
SA_DISPLAY_NAME=anata-no-sa-display-name

$ gcloud iam service-accounts create $SA_NAME \
--display-name $SA_DISPLAY_NAME \
```

<br>

## サービスアカウントに権限を付与

```bash
$ gcloud projects add-iam-policy-binding $PROJECT_ID \
--member serviceAccount:anata-no-sa-name@anata-no-pj-id.iam.gserviceaccount.com \
--role roles/iam.serviceAccountKeyAdmin

$ gcloud projects add-iam-policy-binding $PROJECT_ID \
--member serviceAccount:anata-no-sa-name@anata-no-pj-id.iam.gserviceaccount.com \
--role roles/ml.admin
```

<br>

## サービスアカウントキーの生成

```bash
$ gcloud iam service-accounts keys create titanic_prediction_on_gcp/service_account_keys/key.json \
--iam-account anata-no-sa-name@anata-no-pj-id.iam.gserviceaccount.com
```

<br>

## key.json のパスを環境変数として読み込む

.env ファイルを生成し環境変数とパスを記述

```bash
$ echo GOOGLE_APPLICATION_CREDENTIALS=/content/titanic_prediction_on_gcp/service_account_keys/key.json > /content/titanic_prediction_on_gcp/.env
```

## 確認

```bash
$ cat ./titanic_prediction_on_gcp/.env

# GOOGLE_APPLICATION_CREDENTIALS=/content/titanic_prediction_on_gcp/service_account_keys/key.json
```

<br>
## python-dotenvをインストール
Colabに入っていないので

```bash
$ pip install python-dotenv
```

<br>

## 予測を取得するための関数を定義

```py:notebook
import googleapiclient.discovery
from dotenv import load_dotenv

# 環境変数設定
load_dotenv('/content/titanic_prediction_on_gcp/.env')


def main(input_data):
    input_data = [input_data]

    PROJECT_ID = 'anata-no-pj-id'
    VERSION_NAME = 'v1'
    MODEL_NAME = 'model_xgb'

    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(PROJECT_ID, MODEL_NAME)
    name += '/versions/{}'.format(VERSION_NAME)

    response = service.projects().predict(
        name=name,
        body={'instances': input_data}
    ).execute()

    if 'error' in response:
        print(response['error'])
    else:
        pred = response['predictions'][0]

    return pred
```

<br>

## 年齢と性別のドロップダウンメニューを作成

```py:notebook
import ipywidgets as widgets
from ipywidgets import HBox, VBox


age = [i for i in range(101)]
sex = ['男性', '女性']

dropdown_age = widgets.Dropdown(options=age, description='年齢: ')
dropdown_sex = widgets.Dropdown(options=sex, description='性別: ')
variables = VBox(children=[dropdown_age, dropdown_sex])

VBox(children=[variables])
```

<br>

## 予測

```py:notebook
import numpy as np
from IPython.display import Image
from IPython.display import display_png


input_age = float(dropdown_age.value)
input_sex = 0 if dropdown_sex.value == '男性' else 1
test_input = [input_age, input_sex]

pred = main(test_input)
# print(pred)
pred_binary = np.where(pred > 0.5, 1, 0)
# print(pred_binary)

print('\nあなたがタイタニックに乗ると...')

if pred_binary == 1:
    display_png(Image('/content/titanic_prediction_on_gcp/images/alive.png'))
else:
    display_png(Image('/content/titanic_prediction_on_gcp/images/dead.png'))
```

# 公式リファレンス

https://cloud.google.com/sdk/gcloud/reference/

https://cloud.google.com/sdk/gcloud/reference/ai-platform/

https://cloud.google.com/storage/docs/gsutil

# まとめ

最後まで読んで頂きありがとうございました。
