import torch
import logging
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel
from config import Config

import torch.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel


logger = logging.getLogger(__name__)

cfg = Config()

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
    training_args = TrainingArguments(output_dir=output_path, overwrite_output_dir=True, num_train_epochs=30,
                                      logging_steps=100, save_steps=6000,
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
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", bos_token='<|endoftext|>',
                                              eos_token='<|endoftext|>', pad_token='<|pad|>')
    tokenizer.save_pretrained(cfg.model_gpt2_checkpoint_path)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load the dataset
    with open(cfg.dataset_path_IR_delex, encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    # get 20% test
    total_num_lines = len(lines)
    test_size = int(0.2 * total_num_lines)
    train_val_size = total_num_lines - test_size
    train_val_dataset, test_dataset = random_split(lines, [train_val_size, test_size])

    # generate test dataset
    test_file = open(cfg.dataset_path_test_file, mode='w', encoding='utf-8')
    for row in test_dataset:
        test_file.write(row + "\n")
    test_file.close()

    # formate the datset
    dataset = IROWData(train_val_dataset, tokenizer, cfg.max_length)

    # Training and evaluation
    train(dataset, model, cfg.model_gpt2_checkpoint_path, cfg.model_gpt_neo_log_path)


if __name__ == "__main__":
    main()