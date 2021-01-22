import pandas as pd
from rfi_class import config
from rfi_class.preprocessing.create_dataset import create_dataset, create_folds
from rfi_class.trainer.trainer import model_trainer
from rfi_class.data_loader.data_loader import create_data_loader


def main():

    create_dataset()
    create_folds()

    k_folds = config.N_FOLDS

    df = pd.read_csv(f"{config.BASE_PATH}/data/train_folds.csv")

    for k_fold in range(k_folds):
        df_train = df[df.kfold != k_fold].reset_index(drop=True).head(100)
        df_val = df[df.kfold != k_fold].reset_index(drop=True).head(100)

        train_loader = create_data_loader(df_train, batch_size=config.BATCH_SIZE)
        val_loader = create_data_loader(df_train, batch_size=config.BATCH_SIZE)

        model_trainer(train_loader, val_loader)


if __name__ == "__main__":
    main()