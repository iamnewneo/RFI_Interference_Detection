import glob
import pandas as pd
from pathlib import Path
from rfi_class import config
from sklearn.model_selection import train_test_split, StratifiedKFold


rfi_classes = config.CLASSES
BASE_PATH = config.BASE_PATH


def create_dataset():
    df_list = []
    for class_id, rfi_class in enumerate(rfi_classes):
        df_temp = pd.DataFrame()
        images = glob.glob(f"{BASE_PATH}/data/{rfi_class}/*.png")
        df_temp["path"] = images
        df_temp["path"] = df_temp["path"].astype(str)
        df_temp["target"] = class_id
        df_temp["id"] = df_temp["path"].apply(lambda x: Path(x).stem)
        df_list.append(df_temp)

    df = pd.concat(df_list).reset_index(drop=True)

    X = df[["id", "path"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=config.SEED
    )

    df_train = X_train.copy(deep=True)
    df_train["target"] = y_train
    df_test = X_test.copy(deep=True)
    df_test["target"] = y_test

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    df_train.to_csv(f"{BASE_PATH}/data/train.csv", index=False)
    df_test.to_csv(f"{BASE_PATH}/data/test.csv", index=False)


def create_folds():
    df = pd.read_csv(f"{BASE_PATH}/data/train.csv")
    df = df.sample(frac=1).reset_index(drop=True)
    y = df["target"].values
    kf = StratifiedKFold(n_splits=config.N_FOLDS)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f

    df.to_csv(f"{BASE_PATH}/data/train_folds.csv", index=False)


if __name__ == "__main__":
    create_dataset()
    create_folds()