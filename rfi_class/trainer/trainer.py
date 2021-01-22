import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, early_stopping
from rfi_class import config
from rfi_class.model.model import RFIModel


def model_trainer(train_dataloader, val_dataloader):
    device = config.DEVICE
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="auto"
    )
    model = RFIModel().to(device)
    trainer = pl.Trainer(
        max_epochs=2,
        min_epochs=1,
        callbacks=[early_stop_callback],
        weights_summary=None,
    )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    model_trainer()