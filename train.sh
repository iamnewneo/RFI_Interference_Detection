export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"
python -m rfi_class.driver | tee train_log.log