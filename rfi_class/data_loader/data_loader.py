import multiprocessing
from torch.utils.data import DataLoader
from rfi_class.data_loader.dataset import RFIDataset


def create_data_loader(df, batch_size, resize=None):
    cpu_count = multiprocessing.cpu_count()
    ds = RFIDataset(df=df, resize=resize)
    return DataLoader(ds, batch_size=batch_size, num_workers=cpu_count)