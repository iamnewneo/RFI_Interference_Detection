import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from rfi_class import config
from rfi_class.model.model import RFIModel


def model_trainer(train_dataloader, val_dataloader, progress_bar_refresh_rate):
    device = config.DEVICE
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=5, mode="auto"
    )
    model = RFIModel()
    gpus = None
    if config.DEVICE in ["gpu", "cuda", "cuda:0"]:
        gpus = 1
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=config.MAX_EPOCHS,
        min_epochs=1,
        callbacks=[early_stop_callback],
        weights_summary=None,
        progress_bar_refresh_rate=progress_bar_refresh_rate,
        precision=16,
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    return trainer


if __name__ == "__main__":
    model_trainer()