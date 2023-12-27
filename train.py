import os
import argparse
import wandb

from tqdm import tqdm

import warnings
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Distributed training
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

from config.config import get_config, get_weights_file_path, get_latest_weights_file_path
from dataset.dataset import get_or_build_tokenizer, TextDataset, causal_mask
from model.model import build_transformer

torch.manual_seed(0)

config = get_config()
data_dir = config['data_dir']
train_data_dir = config['train_data_dir']
val_data_dir = config['val_data_dir']
seq_len = config['seq_len']
d_model = config['d_model']
batch_size = config['batch_size']
N = config['N']
h = config['h']
dropout = config['dropout']
d_ff = config['d_ff']
epochs = config['epochs']
lr = config['lr']
model_folder = config['model_folder']
model_basename = config['model_basename']
preload = config['preload']
experiment_name = config['experiment_name']
tokenizer_file = config['tokenizer_file']
force_build_tokenizer = config['force_build_tokenizer']
data_dir = config['data_dir']

tokenizer = get_or_build_tokenizer(tokenizer_file, data_dir, force_build_tokenizer)
vocab_size = tokenizer.get_vocab_size()

train_ds = TextDataset(tokenizer, train_data_dir, seq_len)
val_ds = TextDataset(tokenizer, val_data_dir, seq_len)

# Tensorboard
writer = SummaryWriter(experiment_name)


def train_model():
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                                  sampler=DistributedSampler(train_ds, shuffle=True))

    # Model
    assert torch.cuda.is_available(), "CPU training is not supported"
    device = torch.device("cuda")
    model = build_transformer(vocab_size, seq_len, d_model, N, h, dropout, d_ff).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    initial_epoch = 0
    global_step = 0
    wandb_run_id = None
    if preload is not None:
        if preload == 'latest':
            model_filename = get_latest_weights_file_path(model_folder)
        else:
            model_filename = get_weights_file_path(model_folder, model_basename, int(preload))
        if model_filename is not None:
            print(f"Loading model weights from {model_filename}")
            state = torch.load(model_filename)
            initial_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']
            model.load_state_dict(state['model_state_dict'])
            wandb_run_id = state['wandb_run_id']
            del state
        else:
            print(f"GPU {local_rank} Could not find model weights at {model_filename}")

    # Only initialize W&B on the global rank 0 node
    if global_rank == 0:
        wandb.init(
            # set the wandb project where this run will be logged
            project="pytorch-data-parallel-transformer",
            # allow resuming existing run with the same name (in case the rank 0 node crashed)
            id=wandb_run_id,
            resume="allow",
            # track hyperparameters and run metadata
            config=config
        )
    # Convert the model to DistributedDataParallel
    # Here we can also specify the bucket_cap_mb parameter to control the size of the buckets
    model = DistributedDataParallel(model, device_ids=[local_rank])

    if global_rank == 0:
        # define our custom x axis metric
        wandb.define_metric("global_step")
        # define which metrics will be plotted against it
        wandb.define_metric("validation/*", step_metric="epoch")
        wandb.define_metric("train/*", step_metric="global_step")

    for epoch in range(initial_epoch, epochs):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d} on rank {global_rank}",
                              disable=local_rank != 0)
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)  # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (batch, 1, 1, seq_len)

            # run tensors through transformer
            encoder_output = model.module.encode(encoder_input, encoder_mask)  # (batch, seq_len, d_model)
            proj_output = model.module.project(encoder_output)  # (batch, seq_len, vocab_size)

            label = batch['label'].to(device)  # (batch, seq_len)

            # calculate loss
            # for proj_output -> (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
            # for label -> (batch, seq_len) -> (batch * seq_len)
            loss = loss_fn(proj_output.view(-1, vocab_size), label.view(-1))

            batch_iterator.set_postfix({'loss': loss.item()})

            if global_rank == 0:
                # log the loss to W&B
                wandb.log({'train/loss': loss.item(), 'global_step': global_step})

            # backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        if global_rank == 0:
            # run validation
            run_validation(epoch, model, loss_fn, device)
            test_inference_texts = ['These letters at fit time deliver me',
                                    'I thank thee, Varrius; thou hast made good haste']
            for text in test_inference_texts:
                inference_test(text, model, device)

            # save model after each epoch
            model_filename = get_weights_file_path(model_folder, model_basename, epoch)
            print(f"Saving model weights to {model_filename}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'wandb_run_id': wandb.run.id  # Save to resume logging data
            }, model_filename)


def run_validation(epoch, model, loss_fn, device):
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    model.eval()
    val_loss_accum = 0
    with torch.no_grad():
        batch_iterator_val = tqdm(val_dataloader, desc=f"Validating Epoch {epoch:02d}")
        for batch in batch_iterator_val:
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            encoder_output = model.module.encode(encoder_input, encoder_mask)
            proj_output = model.module.project(encoder_output)
            label = batch['label'].to(device)
            val_loss = loss_fn(proj_output.view(-1, vocab_size), label.view(-1))
            batch_iterator_val.set_postfix({'validation loss': val_loss.item()})
            val_loss_accum += val_loss.item()
        loss = val_loss_accum / len(val_dataloader)
        wandb.log({'validation/loss': loss, 'epoch': epoch})


def inference_test(input_text, model, device):
    model.eval()
    with torch.no_grad():
        input_encoded = tokenizer.encode(input_text).ids
        encoder_input = torch.tensor(input_encoded).unsqueeze(0).to(device)
        while True:
            if encoder_input.size(1) == seq_len:
                break
            encoder_mask = causal_mask(encoder_input.size(1)).type_as(encoder_input).to(device)

            out = model.module.encode(encoder_input, encoder_mask)
            prob = model.module.project(out[:, -1])
            _, next_word = torch.max(prob, dim=-1)
            encoder_input = torch.cat([encoder_input, torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)], dim=1)

        # convert encoder_input to cpu and numpy
        encoder_input = encoder_input.detach().cpu().numpy()
        # decode the tokens
        decoded_tokens = tokenizer.decode(encoder_input[0].tolist())
        # print the decoded tokens
        print(decoded_tokens)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # Disable tokenizers parallelism (this is to avoid deadlocks when creating the tokenizers on multiple GPUs)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Read command line arguments and overwrite config accordingly
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str, default=model_folder)
    args = parser.parse_args()
    config['model_folder'] = args.model_folder
    model_folder = args.model_folder

    # data parallel
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    print(f"Local rank: {local_rank}, Global rank: {global_rank}")

    # Print configuration (only once per server)
    if local_rank == 0:
        print("Configuration:")
        for key, value in config.items():
            print(f"{key:>20}: {value}")

    # Setup distributed training
    init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    train_model()

    # Clean up distributed training
    destroy_process_group()