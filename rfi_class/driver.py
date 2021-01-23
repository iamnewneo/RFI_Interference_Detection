import pandas as pd
from rfi_class import config
from rfi_class.preprocessing.create_dataset import create_dataset, create_folds
from rfi_class.trainer.trainer import model_trainer
from rfi_class.data_loader.data_loader import create_data_loader
from rfi_class.trainer.predict import evaluate_model


def main():

    create_dataset()
    create_folds()

    k_folds = config.N_FOLDS

    df = pd.read_csv(f"{config.BASE_PATH}/data/train_folds.csv")
    df_test = pd.read_csv(f"{config.BASE_PATH}/data/test.csv").head(50)

    test_loader = create_data_loader(df_test, batch_size=config.BATCH_SIZE)

    for k_fold in range(k_folds):
        print(f"Running Fold No: {k_fold}")
        df_train = df[df.kfold != k_fold].reset_index(drop=True).head(10)
        df_val = df[df.kfold != k_fold].reset_index(drop=True).head(10)

        train_loader = create_data_loader(df_train, batch_size=config.BATCH_SIZE)
        val_loader = create_data_loader(df_val, batch_size=config.BATCH_SIZE)

        trainer = model_trainer(train_loader, val_loader, progress_bar_refresh_rate=0)

        model = trainer.get_model()
        result = evaluate_model(model, data_loader=test_loader)
        print(f"Fold {k_fold} Test Loss: {result['loss']:.2f}")
        print(f"Fold {k_fold} Test Accuracy: {result['accuracy']:.2f}")


if __name__ == "__main__":
    main()