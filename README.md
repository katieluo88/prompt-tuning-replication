# The Power of Scale for Parameter-Efficient Prompt Tuning


Re-implementation of prompt tuning for QA, from https://arxiv.org/abs/2104.08691v1 using Pytorch and Huggingface transformers.

## Citation

```bibtex
@misc{lester2021power,
      title={The Power of Scale for Parameter-Efficient Prompt Tuning}, 
      author={Brian Lester and Rami Al-Rfou and Noah Constant},
      year={2021},
      eprint={2104.08691},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Reproduction

All models were trained on the base size for the T5 model, loaded from Hugging Face. The work appears to have use the XXL size model, but due to memory/compute constraints, all reproductions were done on the base sizes (~250 million parameters). Experiments were done on the SQuAD dataset, loaded from Hugging Face.

This work attempted to faithfully reproduce the results from Table 1 of the paper, row 1 (Prompt).

### Soft Prompt Tuning Replication
Run the training for prompt-tuning with the following command:

```bash
./dist_train.sh --output_dir ./prompttune100tok_af_40ep_03lr --prompt_tuning --epochs 40 --optimizer "adafactor" --lr 0.3 --weight_decay 1e-5
```

Primarily a replication project, this work used a few tricks that the work described on the base model size. These include the following:
 1. T5 model from Hugging Face, pre-trained with the LM adaptation objective for up to 100K steps
  * Loaded from Hugging Face: `google/t5-base-lm-adapt`
 2. AdaFactor optimizer with constant learning rate 0.3, and other params as described
 3. Prompt length of 100 tokens, initialized from the vocabulary
 4. Batch size of 32, split between 4 nodes with DDP

One key to the performance of the soft prompt tuning was the learning rate and optimizer. Increasing the learning rate from the default AdaFactor optimizer to 0.3, as described in the work:
| AF Learning Rate | F1 | Exact Match |
| --------- | ------- | -------- |
| 1e-3      |  67.6   |   58.0   |
| 0.3       |  86.0   |   78.6   |


The reimplementation also supports the use of the AdamW optimizer (`optimizer "adam"`) as used by other reimplementations (e.g., PEFT from Hugging Face). However, experimental results show that using the AdaFactor optimizer gives a boost in performance. Results are reported with learning rate of 1e-3 for comparison:

| Optimizer | F1 | Exact Match |
| --------- | ------- | -------- |
| AdamW     |  61.5   |   50.4   |
| AdaFactor |  67.6   |   58.0   |

Experimental challenges included replacing the <pad> tokens in the label with the "ignore" token value of -100, which was common practice but not mentioned explicitly in the paper. In addition, the require batch size of 32 did not fit on even the largest machines, and required either gradient accumulation or distributed training. This work utilizes distributed training across 4 machines, similar to what was described in the paper, using `torch.distributed.run`.

### Final Results

The final results in the reimplementation different slightly from the results in Table 1, most likely due to the difference in model size (base vs. XXL). Final F1 results are below:

| Reimpl? | Model Size | Finetune (Model) | Prompt Tune | Delta |
| ------- | -------- | --------- | ------- | -------- |
| Yes     | Base     |  90.6   |   86.0  | -4.6 |
| No      | XXL      |  94.9   |   94.8  | -0.1 |


### A Note About the Baseline (Fine-tuning)
Run the fine-tuning for the baseline (full-network fine-tuning) with the following command:

```bash
./dist_train.sh --output_dir ./finetune100tok_af_40ep_base --epochs 40 --optimizer "adafactor"
```

Following the work's description, the baseline model is the vanilla T5-base 1.1 version, without LM-adapt step. This was loaded from Hugging Face: `google/t5-v1_1-base`. Testing with and without the LM-adapt seems to yield very little difference in performance:

| LM-adapt? | F1 | Exact Match |
| --------- | ------- | -------- |
| No        |  82.5   |   73.2   |
| Yes       |  82.1   |   72.1   |


However, further experimentation shows that fine-tuning the baseline with a lower learning rate (5e-5) as prescribed by other repo's works significantly better:

| Learning Rate | F1 | Exact Match |
| ----------- | ----------- | ----------- |
| 1e-3      |  82.5   |   73.2   |
| 5e-5      |  90.6   |   83.5   |

These results were on-par with the results reported in the original T5 paper (~82 Exact Match score on SQuAD).

## Concluding Remarks

From this work, one observation is that under the base model size, I was not able to get the prompt-tuning to match the performance of fine-tuning the entire model. The authors did note that the at the XXL size, prompt tuning matches even the stronger multi-task model tuning baseline (section 3.1). One possibility is that there is a performance gap across some or all of the other model sizes; however, this cannot be confirmed as the results for the base size were not reported.

In addition, online implementations including the [official implementation](https://github.com/google-research/prompt-tuning) could not be directly used since it was in JAX, and was mostly used to confirm the hyperparameters used in the final replication. Instead, the project started with utilizing the soft-embedding implementation from [@kipgparker](https://github.com/kipgparker/soft-prompt-tuning). Other QA finetuning [online implementations](https://github.com/zwcolin/Domain-Robustness-Prompt-Tuning) resulted in far lower performance, maybe attributing to the different pre-training, parameters, and optimizer. [PEFT Hugging Face](https://github.com/huggingface/peft) implementation was referenced, but the examples were not on conditional generation (which was required for the task of QA), and utilized a different optimizer than what was reported in the paper, AdamW, which in this experiment showed results in a drop in performance, as well as a significant gap from the final performance. 