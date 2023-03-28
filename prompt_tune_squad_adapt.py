import os
import wandb
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from transformers.optimization import Adafactor, AdafactorSchedule, get_constant_schedule
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import evaluate
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import get_scheduler
from datasets import load_dataset
from soft_embedding import SoftEmbedding
import argparse

from utils import preprocess_function, save_checkpoint
from trainer import train_one_epoch, eval_one_epoch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        required=False,
        help="batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        required=False,
        help="number of epochs to train for",
    )
    parser.add_argument(
        "--n_tokens",
        type=int,
        default=100,
        required=False,
        help="batch size for training",
    )
    parser.add_argument(
        "--prompt_tuning", action="store_true", default=False, help="whether to use prompt tuning"
    )
    parser.add_argument(
        '--optimizer', type=str, default="adafactor", help='optimizer to use'
    )
    parser.add_argument(
        "--lr", type=float,
        default=1e-3, required=False, help="batch size for training",
    )
    parser.add_argument(
        "--weight_decay", type=float,
        default=0, required=False, help="batch size for training",
    )
    parser.add_argument(
        "--dist_train", action="store_true", default=False, help="whether to use prompt tuning"
    )
    parser.add_argument(
        '--output_dir', type=str, default="./default", help='output directory'
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    rank = 0
    if args.dist_train:
        dist.init_process_group("nccl")
        rank = dist.get_rank()

    n_tokens = args.n_tokens
    num_epochs = args.epochs
    batch_size = args.batch_size
    initialize_from_vocab = True

    # ## Handle some loading/logging
    os.makedirs(args.output_dir, exist_ok=True)
    f = open(f"{args.output_dir}/output.txt", "a")  # a bit of a hack to get the output to a file

    # # Get the T5 model
    if args.prompt_tuning:
        # LM-Adapt model
        if rank == 0:
            print("Loading LM-Adapt model")
        tokenizer = T5Tokenizer.from_pretrained("google/t5-base-lm-adapt")
        model = T5ForConditionalGeneration.from_pretrained("google/t5-base-lm-adapt")
    else:
        if rank == 0:
            print("Loading T5-v1_1-base model")
        tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-base")
        model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-base")

    metric = evaluate.load("squad")

    # # Load the SQuAD dataset
    squad_train = load_dataset("squad", split="train")
    squad_test = load_dataset("squad", split="validation")
    # squad_train = load_dataset("squad", split="train[:200]")
    # squad_test = load_dataset("squad", split="validation[:200]")
    # train
    tokenized_squad_train = squad_train.map(lambda examples: preprocess_function(examples, tokenizer=tokenizer, 
                                                                                 n_tokens=n_tokens, prompt_tuning=args.prompt_tuning), 
                                            batched=True, remove_columns=squad_train.column_names)
    tokenized_squad_train.set_format("torch")
    # test
    tokenized_squad_test = squad_test.map(lambda examples: preprocess_function(examples, tokenizer=tokenizer, 
                                                                               n_tokens=n_tokens, prompt_tuning=args.prompt_tuning, 
                                                                               train_set=False), 
                                          batched=True, remove_columns=squad_test.column_names)
    tokenized_squad_test.set_format("torch")

    # ## Set the prompt embeddings
    if args.prompt_tuning:
        # initialize the prompt embeddings with the learned embeddings
        s_wte = SoftEmbedding(model.get_input_embeddings(), 
                              n_tokens=n_tokens, 
                              initialize_from_vocab=initialize_from_vocab)

        model.encoder.set_input_embeddings(s_wte)

        # freeze all parameters except the soft embedding
        for n, p in model.named_parameters():
            if "learned_embedding" not in n:
                # if rank == 0:
                #     print("freezing", n)
                p.requires_grad = False

    device_id = 'cuda'
    if args.dist_train:
        # torch.distributed.init_process_group(backend='nccl')
        # device = torch.device('cuda', torch.distributed.get_rank())
        model.train()
        device_id = rank % torch.cuda.device_count()
        model = model.to(device_id)
        model = DDP(model, device_ids=[device_id])
    else:
        model.cuda()

    if rank == 0:
        wandb.init(project="soft-prompt-tuning",
                   name=f"{args.output_dir.split('/')[-1]}")
        wandb.config.update(args)
        wandb.watch(model)

    # # Train on SQuAD
    train_dataloader = DataLoader(tokenized_squad_train, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(tokenized_squad_test, shuffle=False, batch_size=batch_size)
    if rank == 0:
        print("Number of train elements:", len(train_dataloader))
        print("Number of test elements:", len(test_dataloader))
    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in squad_test]  # test answers 

    if args.optimizer == "adam":
        if rank == 0:
            print("Optimizer: using AdamW")
        optimizer = AdamW(model.parameters(), lr=args.lr)
        num_training_steps = num_epochs * len(tokenized_squad_train)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
    elif args.optimizer == "adafactor":
        if rank == 0:
            print("Optimizer: using Adafactor")
            print("LR:", args.lr)
            print("Weight decay:", args.weight_decay)
        optimizer = Adafactor(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
        )
        # lr_scheduler = AdafactorSchedule(optimizer)
        lr_scheduler = get_constant_schedule(optimizer)
    else:
        raise ValueError("optimizer not recognized")

    model_best_f1 = 0.

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_dataloader, lr_scheduler, device=device_id, rank=rank)        
        squad_output_metrics = eval_one_epoch(model, test_dataloader, theoretical_answers, tokenizer, metric, device=device_id, rank=rank)
        if rank == 0:
            print("--------------------------------------------------")
            print("Epoch", epoch)
            print("Train loss:", train_loss)
            print("Eval metrics:", squad_output_metrics)
            f.write(f'Epoch {epoch} Train loss: {train_loss} Eval metrics: {squad_output_metrics}\n')
        if squad_output_metrics['f1'] > model_best_f1:
            model_best_f1 = squad_output_metrics['f1']
            save_checkpoint(model, optimizer, lr_scheduler, epoch, f'{args.output_dir}/{epoch}_checkpoint.pt', rank)

    # # Test on SQuAD
    squad_output_metrics = eval_one_epoch(model, test_dataloader, theoretical_answers, tokenizer, metric, device=device_id, rank=rank)
    save_checkpoint(model, optimizer, lr_scheduler, epoch, f'{args.output_dir}/last_checkpoint.pt', rank)
    if rank == 0:
        print("SQuAD final metrics:", squad_output_metrics)
        f.write(f'SQuAD final metrics: {squad_output_metrics}\n')
        f.write(f'Save checkpoint to: {args.output_dir}/last_checkpoint.pt')

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        with open(f'{args.output_dir}/results.txt', 'w') as rf:
                    
            for test_qa in squad_test:
                inputs = tokenizer(
                    test_qa["question"],
                    test_qa["context"],
                    max_length=384,
                    truncation="only_second",
                    padding="max_length",
                    return_tensors="pt"
                )
                model_inputs = {}
                model_inputs['input_ids'] = torch.cat([torch.full((inputs['input_ids'].shape[0],n_tokens), 50256), inputs['input_ids']], 1).cuda()
                model_inputs['attention_mask'] = torch.cat([torch.full((inputs['input_ids'].shape[0],n_tokens), 1), inputs['attention_mask']], 1).cuda()
                with torch.no_grad():
                    outputs = model.generate(model_inputs['input_ids'], max_length=20)
                    pred_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                rf.write(f'Question: {test_qa["question"]}\n')
                rf.write(f'Predicted: {pred_answer}\n')
                rf.write(f'True: {test_qa["answers"]["text"]}\n')
                rf.write(f'*******\n')
    f.close()


    # # Finetune on Transfer Task



if __name__ == "__main__":
    main()

