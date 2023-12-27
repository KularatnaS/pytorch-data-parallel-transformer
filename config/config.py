from pathlib import Path

def get_config():

    return {
        'data_dir': 'data/',
        'train_data_dir': 'data/train/',
        'val_data_dir': 'data/val/',
        'force_build_tokenizer': 'false',  # 'true' or 'false'
        'tokenizer_file': 'tokenizer/tokenizer.json',
        # Training parameters
        'seq_len': 32,
        'd_model': 512,
        'd_ff': 1024,
        'N': 6,
        'h': 8,
        'dropout': 0.1,
        'batch_size': 8,
        'lr': 10**-4,
        'epochs': 100,
        # model checkpoint
        'model_folder': 'weights',
        'model_basename': 'tmodel_',
        'preload': 'latest',  # None or epoch number
        'experiment_name': 'runs/tmodel',
    }


def get_weights_file_path(model_folder, model_basename, epoch):
    model_filename = model_basename + str(epoch) + '.pt'
    return str(Path('.') / model_folder / model_filename)


def get_latest_weights_file_path(model_folder):
    # Check all files in the model folder
    model_files = Path(model_folder).glob(f"*.pt")
    # Sort by epoch number (ascending order)
    model_files = sorted(model_files, key=lambda x: int(x.stem.split('_')[-1]))
    if len(model_files) == 0:
        return None
    # Get the last one
    model_filename = model_files[-1]
    return str(model_filename)