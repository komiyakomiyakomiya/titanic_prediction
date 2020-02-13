#%%
import googleapiclient.discovery
# Fill in your PROJECT_ID, VERSION_NAME and MODEL_NAME before running
# this code.

PROJECT_ID = 'mypj-id'
VERSION_NAME = 'v1'
MODEL_NAME = 'model_xgb'

service = googleapiclient.discovery.build('ml', 'v1')
name = 'projects/{}/models/{}'.format(PROJECT_ID, MODEL_NAME)
name += '/versions/{}'.format(VERSION_NAME)

response = service.projects().predict(
    name=name,
    body={'instances': data}
).execute()

if 'error' in response:
  print (response['error'])
else:
  online_results = response['predictions']

print(online_results)



# %%
