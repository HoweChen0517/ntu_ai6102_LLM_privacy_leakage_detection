from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import os
from datasets import Dataset, DatasetDict
import torch

def load_data(dataPath):
    with open(dataPath, 'r', encoding='utf-8') as f:
        data = f.readlines()

    data = [{'label': int(line.strip().split('\t')[0]), 'text': line.strip().split('\t')[1]} for line in data]
    return data

if __name__ == '__main__':
    model_name = '/Volumes/HowesT7/pretrain_models/Qwen2.5-3B-Instruct' # 模型存放位置
    file_path = '../data/'
    save_path = '../checkpoints/Qwen2.5-3B-Instruct/'
    os.makedirs(save_path, exist_ok=True)

    num_train_epochs = 50
    per_device_train_batch_size = 2
    per_device_eval_batch_size = 2
    warmup_steps = 50
    weight_decay = 0.01
    logging_steps = 1
    use_cpu = not torch.cuda.is_available()

    train_data = load_data(os.path.join(file_path, 'train_data.txt'))
    test_data = load_data(os.path.join(file_path, 'test_data.txt'))

    for example in train_data:
        assert example['label'] in [0, 1], f'Invalid label: {example["label"]}'
    for example in test_data:
        assert example['label'] in [0, 1], f'Invalid label: {example["label"]}'

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    # 合并训练集和测试集
    dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})
    print(dataset)
    
    num_labels = 2
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    print(model)
    model.config.pad_token_id = tokenizer.pad_token_id

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True, return_tensors='pt')
    
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # training_args = TrainingArguments(
    #     output_dir=save_path,
    #     num_train_epochs=num_train_epochs,
    #     per_device_train_batch_size=per_device_train_batch_size,
    #     per_device_eval_batch_size=per_device_eval_batch_size,
    #     warmup_steps=warmup_steps,
    #     weight_decay=weight_decay,
    #     logging_steps=logging_steps,
    #     logging_dir='LLM_Privacy_Leakage_Detection',
    #     logging_strategy='epoch',
    #     report_to='wandb',
    #     run_name='Qwen2.5-3B-Instruct',
    #     evaluation_strategy='epoch',
    #     save_strategy='epoch',
    #     save_total_limit=3,
    #     use_cpu=use_cpu
    # )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=encoded_dataset['train'],
    #     eval_dataset=encoded_dataset['test']
    # )

    # print('Train dataset sample:\n', encoded_dataset['train'][0])
    # print('Test dataset sample:\n', encoded_dataset['test'][0])

    # # 开始训练
    # trainer.train()
    # trainer.save_state()
    # trainer.save_model(output_dir=save_path)
    # tokenizer.save_pretrained(save_path)