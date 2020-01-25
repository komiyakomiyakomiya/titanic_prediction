# %%
!pip install --upgrade pip
!pip install autopep8 japanize-matplotlib kaggle python-language-server pyls
!pip install -U pandas==0.25.3

# %%
!mkdir ~/.kaggle
!cp /root/dev/kaggle.json ~/.kaggle
!chmod 600 ~/.kaggle/kaggle.json

print('Done')