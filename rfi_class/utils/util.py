from rfi_class import config


def get_base_path(path=None):
    if path is not None:
        return path
    else:
        return config.BASE_PATH
