# DCI-VTON Checkpoint Download

## Step 1 — Download checkpoint

Google Drive link (from DCI-VTON README):
https://drive.google.com/drive/folders/11BJo59iXVu2_NknKMbN0jKtFV06HTn5K

Download: `viton512.ckpt` (~3.5GB)

## Step 2 — Upload to Kaggle Dataset

1. Go to: https://www.kaggle.com/datasets/new
2. Dataset name: `dci-vton-weights`
3. Upload `viton512.ckpt`
4. Set to Private
5. Create dataset

## Step 3 — Upload notebook

Run:
```cmd
cd C:\vton_project
python ml\kaggle\upload_notebook.py
```
