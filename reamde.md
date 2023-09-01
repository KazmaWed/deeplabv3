動作環境

```
macOS Ventura 13.3.1
Apple M1 Pro
Python 3.11.4
```

ライブラリ

```
pip install torch==1.13.1 \
torchvision==0.14.1 \
torchaudio==0.13.1 \
pytorch_lightning \
torchmetrics==0.6.0 \
pillow \
numpy \
matplotlib
```

ファイル

```
./content/
    ├─ test/
    │    ├─ label/*.png
    │    └─ org/*.jpg
    └─ train/
         ├─ label/*.png
         └─ org/*.jpg
```

参考

https://github.com/insilicomab/semantic_segmentation_chest-Xray_Pytorch-Lightning/blob/main/Xray_pl_deeplabv3_resnet101.ipynb