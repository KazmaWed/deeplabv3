from glob import glob
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torchmetrics
from torchmetrics.functional import accuracy, iou

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3

# ハイパーパラメータの設定
parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4) # 学習率
parser.add_argument('--patience', type=int, default=10) # earlystoppingの監視対象回数
param = parser.parse_args(args=[])

# 画像ファイル名リスト
train_img_list = sorted(glob('./content/train/org/*.jpg'))
test_img_list = sorted(glob('./content/test/org/*.jpg'))
# ラベル画像リスト
train_label_list = sorted(glob('./content/train/label/*.png'))
test_label_list = sorted(glob('./content/test/label/*.png'))

# データセットクラス
class XrayDataset(data.Dataset):
    def __init__(self, img_path_list, label_path_list):
        self.image_path_list = img_path_list
        self.label_path_list = label_path_list
        self.transform = transforms.Compose( [transforms.Resize((param.image_size, param.image_size)),
                                              transforms.ToTensor(),])
    def __len__(self):
        return len(self.image_path_list)

    
    def __getitem__(self, idx):
        img = Image.open(self.image_path_list[idx]).convert('RGB')
        img = self.transform(img)
        label = Image.open(self.label_path_list[idx])
        label = self.transform(label)
        return img, label

# ネットワークの定義と訓練
class Net(pl.LightningModule):
    def __init__(self, lr: float):
        super().__init__()
        self.lr = lr
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        # self.model.classifier = deeplabv3.DeepLabHead(2048, 1)
        self.model.classifier = deeplabv3.DeepLabHead(2048, 3)

    def forward(self, x):
        h = self.model(x)
        return h

    def training_step(self, batch, batch_idx):
        x, t = batch
        out = self(x)
        y = torch.sigmoid(out['out'])
        loss = F.binary_cross_entropy_with_logits(out['out'], t)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', accuracy(y, t.int()), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou(y, t.int()), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        out = self(x)
        y = torch.sigmoid(out['out'])
        loss = F.binary_cross_entropy_with_logits(out['out'], t)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy(y, t.int()), on_step=False, on_epoch=True)
        self.log('val_iou', iou(y, t.int()), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
    
if __name__ == '__main__':
    print("Pythonのバージョン：",sys.version)
    print("PyTorchのバージョン：", torch.__version__)
    print(param)

    # 保存先のディレクトリを作成する
    SAVE_MODEL_PATH = './model/'  # モデルの保存先
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    # Datasetのインスタンス作成
    train_dataset = XrayDataset(train_img_list, train_label_list)
    test_dataset = XrayDataset(test_img_list, test_label_list)

    # datasetの確認
    print(f'len(train_dataset): {len(train_dataset)}, len(test_dataset): {len(test_dataset)}')

    # 1 サンプル目の入力値と目標値を取得
    x, t = train_dataset[0]
    # imageのshapeを確認
    print(f'x.shape: {x.shape}, t.shape: {t.shape}')

    # Dataloader
    dataloader = {
        'train': data.DataLoader(train_dataset, batch_size=param.batch_size, shuffle=True),
        'val': data.DataLoader(test_dataset, batch_size=param.batch_size, shuffle=False)
    }

    # callbacksの定義
    model_checkpoint = ModelCheckpoint(
        SAVE_MODEL_PATH,
        filename="DeepLabV3_resnet101"+"{epoch:02d}-{val_loss:.2f}",
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=False,
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=param.patience,
    )

    # 訓練の実行
    pl.seed_everything(0)
    net = Net(lr=param.lr)
    trainer = pl.Trainer(max_epochs=param.epochs, 
                        callbacks=[model_checkpoint, early_stopping],
                        accelerator='cpu',
                        gpus=1)
    trainer.fit(net, dataloader['train'], dataloader['val'])
     
    # 訓練データと検証データに対する最終的な結果を表示
    trainer.callback_metrics

    # 結果の可視化
    n_max_imgs = 5

    plt.figure(figsize=(9, 20))
    for n in range(n_max_imgs):
        x, t = test_dataset[n]
        net.eval()
        out = net(x.unsqueeze(0)) # 最初の位置（0）に新たな次元を挿入
        y = torch.sigmoid(out['out'])
        y_label = (y > 0.5).int().squeeze()

        t = np.squeeze(t) # numpyでtensorの余分な次元を除去

        plt.subplot(n_max_imgs, 2, 2*n+1)
        plt.imshow(t.T, cmap='hsv')

        plt.subplot(n_max_imgs, 2, 2*n+2)
        plt.imshow(y_label.T, cmap='hsv')
        
        plt.show()