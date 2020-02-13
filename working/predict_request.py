# %%
import googleapiclient.discovery

from data import Data


def main():
    data = Data()
    train_x, train_y, test_x = data.processing()

    input_data = [[36, 0]]

    print(input_data)

    PROJECT_ID = 'mypj-id'
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
        online_results = response['predictions']

    print(response)
    print(online_results)

    # print(online_results)


if __name__ == '__main__':
    main()


# %%
