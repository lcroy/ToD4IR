import torch
import os
from torch import nn
import logging
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPTNeoForCausalLM
from config import Config
from tqdm import tqdm, trange

import torch.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

# we train it on 4 gpus
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)

torch.manual_seed(22)


class IROWData(Dataset):
    def __init__(self, lines, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for line in lines:
            encodings_dict = tokenizer(line, truncation=True,
                                       max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


def train(dataset, model, output_path, log_path):
    # split the dataset to training and validation
    total_num_dataset = len(dataset)
    train_size = int(0.8 * total_num_dataset)
    val_size = total_num_dataset - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(len(train_dataset), len(val_dataset))
    # set up the training arguments
    training_args = TrainingArguments(output_dir=output_path, overwrite_output_dir=True, num_train_epochs=50,
                                      logging_steps=10, save_strategy='epoch',
                                      per_device_train_batch_size=1, per_device_eval_batch_size=1,
                                      warmup_steps=100, weight_decay=0.01, logging_dir='./logs',
                                      logging_strategy='epoch', evaluation_strategy="epoch")

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                  'attention_mask': torch.stack([f[1] for f in data]),
                                                  'labels': torch.stack([f[0] for f in data])})
    trainer.train()


def main():
    cfg = Config()
    # initial gpt-neo model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", bos_token='<|endoftext|>',
                                              eos_token='<|endoftext|>', pad_token='<|pad|>')
    tokenizer.save_pretrained(cfg.model_gpt_neo_checkpoint_path)
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model.resize_token_embeddings(len(tokenizer))

    # train parallel on 4 gpus
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    # # torch.distributed.init_process_group(backend="nccl")
    # model = DistributedDataParallel(model)
    # model = model.cuda()
    # model = nn.parallel.DistributedDataParallel(model)
    # model = nn.DataParallel(model)
    # model = model.cuda()
    # train single
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load the dataset
    with open(cfg.dataset_path_IR_delex, encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    # formate the datset
    dataset = IROWData(lines, tokenizer, cfg.max_length)

    # Training and evaluation
    train(dataset, model, cfg.model_gpt_neo_checkpoint_path, cfg.model_gpt_neo_log_path)


if __name__ == "__main__":
    main()