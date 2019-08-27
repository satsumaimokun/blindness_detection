#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


cd /content/drive/My Drive/aptos2019-blindness-detection


# In[19]:


# パッケージのimport
from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms


# In[20]:


# 乱数のシードを設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


# In[21]:


# 入力画像の前処理をするクラス(切り取るだけ)

class ImageTransform():

    def __init__(self, resize=256):
        self.data_transform = transforms.Compose([
                transforms.Resize(resize),  # リサイズ
                transforms.CenterCrop(resize),  # 画像中央をresize×resizeで切り取り
                transforms.ToTensor(),  # テンソルに変換
            ])

    def __call__(self, img):
        return self.data_transform(img)


# In[22]:


# 画像へのファイルパスのリストを作成する

def make_datapath_list(phase="train"):
    """
    データのパスを格納したリストを作成する。

    Parameters
    ----------
    phase : 'train' or "test"
        訓練データ(val用のデータも含む)かテストデータかを指定する

    Returns
    -------
    path_list : list
        データへのパスを格納したリスト
    """

    rootpath = "./"
    target_path = osp.join(rootpath+phase+"_images/*.png") #ここにimagesのファイルがある前提
    print(target_path)

    path_list = []  # ここに格納する

    # globを利用してサブディレクトリまでファイルパスを取得する
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


# 実行
train_and_val_list = make_datapath_list(phase="train")
#test_list = make_datapath_list(phase="test")

#train_and_val_list


# In[23]:


# 訓練データ(traindata)と検証データ(valdata)をtrain_test_splitで分けてtrain_listとval_listを作る
train_csv_file = "train.csv"
df = pd.read_csv(train_csv_file)

id_code = df["id_code"]
diagnosis = df["diagnosis"]

# train_test_split(stratify=diagnosisで層化抽出)
train_id_code, _, _, val_labels = train_test_split(
    id_code, diagnosis, test_size=0.3, random_state=1234, stratify=diagnosis)

train_list=[]
val_list=[]

# trainのリスト
for train_id in train_id_code:
    for file_path in train_and_val_list:
        if str(train_id) in file_path:
            train_list.append(file_path)
            
# valのリスト
for file_path in train_and_val_list:
    if (file_path in train_list) == False:
        val_list.append(file_path)
        
# 確認
print(len(train_list))
print(len(val_list))
print(len(train_and_val_list))
print(len(train_list)/len(train_and_val_list))


# In[24]:


print(diagnosis.value_counts())
print(diagnosis.mean())


# In[25]:


# 画像のDatasetを作成する


class RetinopathyDataset(data.Dataset):
    """
    画像のDatasetクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    file_list : リスト
        画像のパスを格納したリスト
    transform : object
        前処理クラスのインスタンス
    phase : 'train' or 'test'
        学習かテストかを設定する。
    """

    def __init__(self, file_list, transform=None, phase='train', csv_file=None): #現状phaseはついてるだけ
        self.file_list = file_list  # ファイルパスのリスト
        self.transform = transform  # 前処理クラスのインスタンス
        self.phase = phase  # train or testの指定
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        '''画像の枚数を返す''' 
        return len(self.file_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとラベルを取得
        '''

        # index番目の画像をロード
        img_path = self.file_list[index]
        img = Image.open(img_path)  # [高さ][幅][色RGB]

        # 画像の前処理を実施
        img_transformed = self.transform(
            img)  # torch.Size([3, 224, 224])

        # 画像のラベル
        label = self.data.at[index, "diagnosis"]
        if label == 0:
            label = -10 #0を離れたところに置いてみる
        
        return img_transformed, label


# 実行
train_dataset = RetinopathyDataset(
    file_list=train_list, transform=ImageTransform(), phase='train',csv_file="train.csv")

val_dataset = RetinopathyDataset(
    file_list=val_list, transform=ImageTransform(), phase='val',csv_file="train.csv")

#test_dataset = RetinopathyDataset(
#    file_list=test_list, transform=ImageTransform(), phase='test',csv_file="test.csv")

# 動作確認
index = 0
print(train_dataset.__getitem__(index)[0].size())
print(train_dataset.__getitem__(index)[1])


# #(色、高さ、幅)を (高さ、幅、色)に変換して表示
# img_transformed = train_dataset.__getitem__(index)[0]
# img_transformed = img_transformed.numpy().transpose((1, 2, 0))
# img_transformed = np.clip(img_transformed, 0, 1) #色をいじった時は0,1に直して表示
# plt.imshow(img_transformed)
# plt.show()


# In[26]:


# ミニバッチのサイズを指定
batch_size = 16

# DataLoaderを作成
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True) #num_workers=4にするとBroken pipeと言われた

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False) #num_workers=4にするとBroken pipeと言われた

#test_dataloader = torch.utils.data.DataLoader(
#    test_dataset, batch_size=batch_size, shuffle=False)

#辞書型変数にまとめる
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

# 動作確認
batch_iterator = iter(dataloaders_dict["train"])  # イテレータに変換
inputs, labels = next(batch_iterator)  # 1番目の要素を取り出す
print(inputs.size())
print(labels)


# In[27]:


# 学習済みのresnet101モデルをロード
# resnet101モデルのインスタンスを生成
use_pretrained = True  # 学習済みのパラメータを使用
net = models.resnet101(pretrained=use_pretrained)

# resnet101の最後の出力層の出力ユニットを回帰の1つに取り換える
net.fc = nn.Linear(in_features=2048, out_features=1)

# 訓練モードに設定
net.train()

print('ネットワーク設定完了：学習済みの重みをロードし、訓練モードに設定しました')


# In[3]:


# #重みをロードする
# load_path = "./weights_basemodel.pth"
# load_weights=torch.load(load_path)

# #networkを立てる
# net = models.resnet101(pretrained=False)
# net.fc = nn.Linear(in_features=2048, out_features=1)
# net.load_state_dict(load_weights)

# # 訓練モードに設定
# net.train()

# print('ネットワーク設定完了：学習済みの重みをロードし、訓練モードに設定しました')


# In[28]:


# モデルを見てみる
#print(models.resnet101(pretrained=True))


# In[29]:


# 損失関数の設定
criterion = nn.MSELoss()


# In[30]:


# 学習させるパラメータを、変数params_to_updateの1,2に格納する

params_to_update_1 = []
params_to_update_2 = []

# 学習させる層のパラメータ名を指定(layer4に含まれる層と最後の層のパラメタをいじる)
update_param_names_1=[]
for name, _ in net.layer4.named_parameters():
    update_param_names_1.append("layer4."+name)
update_param_names_2 = ["fc.weight", "fc.bias"]

# パラメータごとに各リストに格納する
for name, param in net.named_parameters():

    if name in update_param_names_1:
        param.requires_grad = True
        params_to_update_2.append(param)
        #print("params_to_update_1に格納：", name)

    elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
        #print("params_to_update_2に格納：", name)

    else:
        param.requires_grad = False #とりあえず他のやつは勾配計算しない
        #print("勾配計算なし。学習しない：", name)


# In[31]:


# 最適化手法の設定
optimizer = optim.Adam([
    {'params': params_to_update_1, 'lr': 1e-4},
    {'params': params_to_update_2, 'lr': 1e-3},
], lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4) #4epochsでlrを1/10倍


# In[32]:


# モデルを学習させる関数を作成

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # 初期設定
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # epochのループ
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')
        
        scheduler.step() #schedulerのstepに+1
        
        #trainとvalのループ
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   #　モデルを推論モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_acc = 0 # epochの精度
            
            # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
            if (epoch == 0) and (phase == 'train'):
                continue

            # データローダーからミニバッチを取り出すループ
            for inputs, labels in tqdm(dataloaders_dict[phase]):

                # GPUが使えるならGPUにデータを送る
                inputs = inputs.to(device)
                labels = labels.to(device)

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    labels = labels.view(-1,1) #整形
                    labels = labels.float() #型をfloatに
                    loss = criterion(outputs, labels)  # 損失を計算
                    
                    # 訓練時はバックプロパゲーション
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    # lossの合計を更新
                    epoch_loss += loss.item() * inputs.size(0) 
                    # 精度の計算
                    pred =  torch.zeros(outputs.size()[0], 1)
                    pred = pred.to(device)
                    n = 0
                    for i in outputs:
                        if i<0.5:
                            pred[n,0] = -10
                        elif i<1.5:
                            pred[n,0] = 1
                        elif i<2.5:
                            pred[n,0] = 2
                        elif i<3.5:
                            pred[n,0] = 3
                        else:
                            pred[n,0] = 4
                            
                        n += 1
                        
                    diff = labels - pred
                    for i in diff:
                        if i == 0:
                            epoch_acc += 1

            # epochごとのaccとlossを表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_acc / len(dataloaders_dict[phase].dataset)
            print("{} Loss: {:.4f}".format(phase, epoch_loss))
            print("{} acc: {:.4f}".format(phase, epoch_acc))


# In[33]:


# 学習・検証を実行する
num_epochs=13
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)


# In[44]:


#重みを保存する
save_path = "./weights_basemodel.pth"
torch.save(net.state_dict(),save_path)


# In[51]:


#重みをロードする
load_path = "./weights_basemodel.pth"
load_weights=torch.load(load_path)

#networkを立てる
net = models.resnet101(pretrained=False)
net.fc = nn.Linear(in_features=2048, out_features=1)
net.load_state_dict(load_weights)

