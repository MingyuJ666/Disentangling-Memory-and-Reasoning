import random
from typing import Dict, Optional, Sequence
import torch
from torch.utils.data import IterableDataset
import transformers
from load_data.supervised_dataset import SupervisedDataset


class ConstantLenDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            tokenized (bool): If true we use a pretokenized dataset.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
    ):  
        super(ConstantLenDataset, self).__init__()
        self.data = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.epoch = 0
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.num_of_sequences = num_of_sequences
    
    def set_epoch(self, epoch):
        random.seed(epoch)

    def iter_fun(self):
        ids = list(range(len(self.data)))
        random.shuffle(ids)
        for i in ids:
            sent = self.data[i]['x'] + self.data[i]['y']
            yield sent

    # def __len__(self):
    #     return len(self.data)//self.num_of_sequences

    def __iter__(self):
        more_examples = True
        iterator = self.iter_fun()
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    iterator = self.iter_fun()
                    self.epoch += 1
                    print(f"Dataset epoch: {self.epoch}")
            
            random.shuffle(buffer)
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.tokenizer.eos_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    yield dict(input_ids=torch.tensor(input_ids), labels=torch.tensor(input_ids))
                    

def make_constant_len_data_module(tokenizer: transformers.PreTrainedTokenizer, 
                                dataset, eval_dataset, seq_length) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    random.seed(42)
    if eval_dataset is not None and len(eval_dataset) > 1000:
        idx = random.choices(list(range(len(eval_dataset))), k=1000)
        new_x = []
        new_y = []
        for i in idx:
            new_x.append(eval_dataset[i]['x'])
            new_y.append(eval_dataset[i]['y'])
        eval_dataset.x = new_x
        eval_dataset.y = new_y
    assert len(eval_dataset) <= 1000
    train_dataset = ConstantLenDataset(dataset, tokenizer, seq_length)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)