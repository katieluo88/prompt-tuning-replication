import torch


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def save_checkpoint(model, optimizer, scheduler, epoch, experiment_path, rank=0):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_state = model_state_to_cpu(model.module.state_dict())
    else:
        model_state = model.state_dict()
    if rank == 0:
        print("Saving checkpoint...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }, experiment_path)


def preprocess_function(examples, tokenizer, n_tokens=20, train_set=True, prompt_tuning=False):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_tensors="pt"
    )

    # model_inputs = {}

    # pad the inputs if prompt tuning
    if prompt_tuning:
        inputs['input_ids'] = torch.cat([torch.full((inputs['input_ids'].shape[0],n_tokens), 50256), inputs['input_ids']], 1)
        inputs['attention_mask'] = torch.cat([torch.full((inputs['input_ids'].shape[0],n_tokens), 1), inputs['attention_mask']], 1)
    else:
        inputs['input_ids'] = inputs['input_ids']
        inputs['attention_mask'] = inputs['attention_mask']
    
    if train_set:
        answers = [a['text'][0].strip() for a in examples["answers"]]
        targets = tokenizer(
            answers,
            max_length=384,
            truncation="only_second",
            padding="max_length",
            return_tensors="pt"
        )
        
        inputs['labels'] = targets['input_ids']
        # print("before:", model_inputs['labels'])
        inputs['labels'][inputs['labels'] == tokenizer.pad_token_id] = -100
    else:
        inputs['id'] = examples['id']

    #     model_inputs['answers'] = examples['answers']
    
    return inputs