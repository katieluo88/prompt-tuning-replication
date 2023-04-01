import wandb
import torch
from tqdm import tqdm


def train_one_epoch(model, optimizer, train_dataloader, lr_scheduler, device, rank=0, grad_accum_steps=1):
    model.train()

    if rank == 0:
        pbar = tqdm(total=len(train_dataloader), leave=False,
                         desc='train', dynamic_ncols=True, disable=False)
    
    epoch_loss = 0
    # import ipdb; ipdb.set_trace()
    for idx, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / grad_accum_steps
        loss.backward()
        if (idx + 1) % grad_accum_steps == 0 or (idx + 1) == len(train_dataloader):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        epoch_loss += loss.item()
        if rank == 0:
            pbar.update()
            wandb.log({"train_loss": loss.item()})
    if rank == 0:
        pbar.close()
    return epoch_loss / len(train_dataloader)


def eval_one_epoch(model, test_dataloader, labels, tokenizer, metric, device, rank=0):
    model.eval()

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    if rank == 0:
        pbar = tqdm(total=len(test_dataloader), leave=False,
                         desc='val', dynamic_ncols=True, disable=False)
        
    predicted_answers = []
    for batch in test_dataloader:
        # batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device), 
                attention_mask=batch["attention_mask"].to(device), max_length=30)
            # tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            pred_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for example_id, best_answer in zip(batch['id'], pred_answers):
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer}
            )
        if rank == 0:
            pbar.update()
    squad_output_metrics = metric.compute(predictions=predicted_answers, references=labels)
    if rank == 0:
        pbar.close()
        wandb.log(squad_output_metrics)
    return squad_output_metrics