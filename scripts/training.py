from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_scheduler
)
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler
import click
import torch


@click.command()
@click.option('--checkpoint', default='sshleifer/tiny-gpt2', help='Model')
@click.option('--batch_size', default=2, help='Batch size')
@click.option('--num_warmup_steps', default=1, help='Warmup')
@click.option('--total_num_steps', default=1, help='Warmup')
@click.option('--num_train_epochs', default=10, help='epochs')
def train(checkpoint, batch_size, num_warmup_steps, total_num_steps, num_train_epochs):
    accelerator = Accelerator(device_placement=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    eli5 = load_dataset("eli5", split="train_asks[:5000]")

    def preprocess_function(row):
        return tokenizer(row['title'] + '\n' + row['answers']['text'][0], return_tensors='pt', padding="max_length",
                         max_length=256, truncation=True)

    train_dataset = eli5.map(
        preprocess_function,
        batched=False,
        num_proc=1,
        remove_columns=['q_id', 'title', 'selftext', 'document', 'subreddit', 'answers', 'title_urls', 'selftext_urls',
                        'answers_urls']
    )

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
           or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=1e-4)

    max_train_steps = 10_000

    if (
            accelerator.state.deepspeed_plugin is None
            or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=10_000,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=max_train_steps, warmup_num_steps=num_warmup_steps
        )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    for epoch in range(0, num_train_epochs):
        model.train()

        for step, batch in tqdm(enumerate(train_dataloader)):

            with accelerator.accumulate(model):
                outputs = model(input_ids=batch['input_ids'], labels=batch['input_ids'])
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


if __name__ == '__main__':
    train()
