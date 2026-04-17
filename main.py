import math
import os
import time
from tempfile import TemporaryDirectory
import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn

from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from model import TransformerModel
from data_process import data_process, get_batch, batchify
from model_util import positionalEncodingTest


def main():
    # Configuration
    batch_size = 20         # Batch size for training
    eval_batch_size = 10    # Batch size for evaluation
    seq_len = 35            # Sequence length
    emsize = 200            # Embedding dimension
    d_hid = 200             # Dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2             # Number of transformer encoder layer in nn.TransformerEncoder
    nhead = 2               # Number of attention heads in nn.MultiheadAttention
    dropout = 0.2           # Dropout probability
    lr = 5.0                # Learning rate
    epochs = 3              # Training epoch

    # Get PennTreebank Dataset & Tokenizer
    train_iter = PennTreebank(split='train')
    tokenizer = get_tokenizer('basic_english')

    # Build Vocab
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    rev_vocab_dict = dict(enumerate(vocab.get_itos()))

    # Split Dataset: Train / Validation / Test
    train_iter, val_iter, test_iter = PennTreebank()

    print("## First 5 sentences of the training dataset: ")
    print('\n'.join([t for t in train_iter][:5]))
    print()

    # Tokenize the datasets
    train_data = data_process(train_iter, vocab, tokenizer)
    val_data = data_process(val_iter, vocab, tokenizer)
    test_data = data_process(test_iter, vocab, tokenizer)

    # Torch will use GPU if your computer has one, otherwise CPU. (GPU programming may require some other packages)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare the datasets for mini-batch training.
    train_data = batchify(train_data, batch_size, device)  # shape [seq_len, batch_size]
    val_data = batchify(val_data, eval_batch_size, device)
    test_data_for_prediction = batchify(test_data, 1, device)
    test_data = batchify(test_data, eval_batch_size, device)

    ntokens = len(vocab)  # size of vocabulary

    positionalEncodingTest()  # Q1

    print("\n## Q2: Building Transformer Encoder using PyTorch (60 pts)")
    # Build Transformer model
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout, device).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # Optimizer: Stochastic Gradient Descent
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)  # Learning rate decays by gamma every epoch.

    best_val_loss = float('inf')

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            # Train the model for an epoch
            train(model, train_data, optimizer, scheduler, criterion, seq_len, ntokens, epoch)

            # Evaluate the model with validation data
            val_loss = evaluate(model, val_data, seq_len, ntokens, criterion)
            val_ppl = math.exp(val_loss)

            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                  f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
            print('-' * 89)

            # The model state is saved whenever the validation loss is the lowest.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)

            scheduler.step()

        # Load the best model states
        model.load_state_dict(torch.load(best_model_params_path))

        # Evaluate the performance on the test(unseen) dataset
        test_loss = evaluate(model, test_data, seq_len, ntokens, criterion)
        test_ppl = math.exp(test_loss)
        print('=' * 89)
        print(f'| End of training | test loss {test_loss:5.2f} | '
              f'test ppl {test_ppl:8.2f}')
        print('=' * 89)

        if test_ppl < 200.0:
            print("\nCongratulations! You have completed all tasks.")
        else:
            print(f'\nFAIL. Your test ppl is {test_ppl}, while your test ppl should be below 200.')
            exit(1)


        # Check the next-word-prediction examples
        predict(model, test_data_for_prediction, seq_len, ntokens, rev_vocab_dict)


def train(model, train_data, optimizer, scheduler, criterion, seq_len, ntokens, epoch):
    model.train()  # turn on train mode

    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = len(train_data) // seq_len

    # Train the model with each batch
    for batch, i in enumerate(range(0, train_data.size(0) - 1, seq_len)):
        data, targets = get_batch(train_data, i, seq_len)

        # Forward propagation
        output = model(data)
        output_flat = output.view(-1, ntokens)

        # Get loss with the loss function
        loss = criterion(output_flat, targets)

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


def evaluate(model, eval_data, seq_len, ntokens, criterion):
    model.eval()  # turn on evaluation mode (for deactivating dropout, ...)
    total_loss = 0.
    with torch.no_grad():  # No gradient is generated in this scope
        for i in range(0, eval_data.size(0) - 1, seq_len):
            data, targets = get_batch(eval_data, i, seq_len)
            seq_len = data.size(0)
            output = model(data)  # B * l * V
            output_flat = output.view(-1, ntokens)

            # Get validnation (or test) loss
            total_loss += seq_len * criterion(output_flat, targets).item()

    return total_loss / (len(eval_data) - 1)


def predict(model, eval_data, seq_len, ntokens, rev_vocab_dict):
    model.eval()  # turn on evaluation mode (for deactivating dropout, ...)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, seq_len):
            data, targets = get_batch(eval_data, i, seq_len)

            output = model(data)
            output_flat = output.view(-1, ntokens)

            # Get predicted next word (index with the maximum prediction score)
            prediction = torch.argmax(output_flat, dim=-1)

            print(f"Input Tokens: {' '.join([rev_vocab_dict[d] for d in data.view(-1).tolist()])}")

            print(f"Predicted Next Word: {rev_vocab_dict[prediction[seq_len-1].item()]}")
            input("Press Enter to see the next example\n")


if __name__ == '__main__':
    main()
