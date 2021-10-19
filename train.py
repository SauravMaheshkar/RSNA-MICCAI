import pandas as pd
import torch
import wandb
from sklearn import model_selection as sk_model_selection
from torch.nn import functional as torch_functional
from torch.utils import data as torch_data

from rsna.dataloader import Dataset
from rsna.engine import Trainer
from rsna.nn import Model

mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
train_df = pd.read_csv(f"{data_directory}/train_labels.csv")
df_train, df_valid = sk_model_selection.train_test_split(
    train_df, test_size=0.2, random_state=12, stratify=train_df["MGMT_value"]
)


def train_mri_type(df_train, df_valid, mri_type):
    if mri_type == "all":
        train_list = []
        valid_list = []
        for mri_type in mri_types:
            df_train.loc[:, "MRI_Type"] = mri_type
            train_list.append(df_train.copy())
            df_valid.loc[:, "MRI_Type"] = mri_type
            valid_list.append(df_valid.copy())

        df_train = pd.concat(train_list)
        df_valid = pd.concat(valid_list)
    else:
        df_train.loc[:, "MRI_Type"] = mri_type
        df_valid.loc[:, "MRI_Type"] = mri_type

    print(df_train.shape, df_valid.shape)

    train_data_retriever = Dataset(
        df_train["BraTS21ID"].values,
        df_train["MGMT_value"].values,
        df_train["MRI_Type"].values,
        augment=True,
    )

    valid_data_retriever = Dataset(
        df_valid["BraTS21ID"].values,
        df_valid["MGMT_value"].values,
        df_valid["MRI_Type"].values,
    )

    train_loader = torch_data.DataLoader(
        train_data_retriever, batch_size=4, shuffle=True, num_workers=8
    )

    valid_loader = torch_data.DataLoader(
        valid_data_retriever, batch_size=4, shuffle=False, num_workers=8
    )

    run = wandb.init(project="RSNA-MICCAI", entity="sauravmaheshkar")

    model = Model()
    model.to(device)

    wandb.watch(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    criterion = torch_functional.binary_cross_entropy_with_logits

    trainer = Trainer(model, device, optimizer, criterion)

    history = trainer.fit(  # noqa: F841
        10, train_loader, valid_loader, f"{mri_type}", 10
    )

    run.finish()

    return trainer.lastmodel


modelfiles = None


if not modelfiles:
    modelfiles = [train_mri_type(df_train, df_valid, m) for m in mri_types]
    print(modelfiles)
