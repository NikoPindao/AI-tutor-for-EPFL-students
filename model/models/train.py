from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig, setup_chat_format
import os


model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
RANDOM_SEED= 42
#model_name = "microsoft/Phi-3-mini-128k-instruct"
bnb_config = BitsAndBytesConfig(
 load_in_4bit=False,
 bnb_4bit_quant_type="nf4",
 bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config)

peft_config = LoraConfig(r=32, lora_alpha=64, target_modules='all-linear', lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
base_model = get_peft_model(base_model, peft_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

base_model, tokenizer = setup_chat_format(base_model, tokenizer)
base_modelmodel = prepare_model_for_kbit_training(base_model)

tokenizer.pad_token = tokenizer.eos_token
#path_to_data = '../../data/DPO_Dataset/final_data_dpo.jsonl'
path_to_data = "/home/moteloumka/EPFL/MA2/MNLP/project-m2-2024-nnnandanders/data/DPO_Dataset/final_data_dpo_m3.jsonl"
dataset = load_dataset('json', data_files=path_to_data,split='train').train_test_split(test_size=0.01, seed=RANDOM_SEED)


def format_chat_template(row):
    #transform the input data into the format Llama 3 requires 
    row["prompt"] = {"role": "user", "content": row["chosen"]}
    row["chosen"] = {"role": "assistant", "content": row["chosen"]}
    row["rejected"] = {"role": "assistant", "content": row["rejected"]}
    #let the tokeniser do the preprocessing
    row["prompt"] = tokenizer.apply_chat_template([row["prompt"]], tokenize=False)
    row["chosen"] = tokenizer.apply_chat_template([row["chosen"]], tokenize=False)
    row["rejected"] = tokenizer.apply_chat_template([row["rejected"]], tokenize=False)

    return row

dataset = dataset.map(
    format_chat_template,
    num_proc= os.cpu_count(),
)


train_dataset = dataset['train']
#eval_dataset = load_dataset('json', data_files=path_to_data,split='train').train_test_split(test_size=0.002)
eval_dataset = dataset['test']

args = DPOConfig(
    output_dir="../checkpoints",               # directory to save and repository id
    num_train_epochs=2,                     # number of training epochs
    per_device_train_batch_size=2,         # batch size per device during training
    per_device_eval_batch_size=1,           # batch size for evaluation
    gradient_accumulation_steps=5,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    learning_rate=5e-5,                     # 10x higher LR than QLoRA paper
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
    lr_scheduler_type="cosine",             # use cosine learning rate scheduler
    logging_steps=25,                       # log every 25 steps
    save_steps=1000,                         # when to save checkpoint
    save_total_limit=60,                     # limit the total amount of checkpoints
    evaluation_strategy="steps",            # evaluate every 1000 steps
    eval_steps=1000,                         # when to evaluate
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    push_to_hub=False,                      # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
)
 
dpo_args = {
    "beta": 0.1,                            # The beta factor in DPO loss. Higher beta means less divergence
    "loss_type": "sigmoid"                  # The loss type for DPO.
}

max_seq_length = 1024
prompt_length = 1024

trainer = DPOTrainer(
    base_model,
    ref_model=None, # set to none since we use peft
    peft_config=peft_config,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_length=max_seq_length,
    max_prompt_length=prompt_length,
    beta=dpo_args["beta"],
    loss_type=dpo_args["loss_type"]
)

trainer.train()

