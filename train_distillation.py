import os
import sys
import pandas as pd
import numpy as np
import torch
from utilities.data import VLM_dataset
from utilities.utils import Arguments, DistillationTrainer, DistillationTrainingArguments
from transformers import AutoTokenizer, AutoImageProcessor,AutoConfig, AutoModelForSequenceClassification, AutoModelForImageClassification,EarlyStoppingCallback
from transformers import set_seed, HfArgumentParser
from torch.utils.data import DataLoader

def collate_fn(examples):
    pixel_values = torch.cat([example["pixel_values"] for example in examples], dim=0)
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    labels = torch.tensor([example["labels"] for example in examples])
    
    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "attention_mask": attention_mask,
        "labels": labels,
        "return_loss": True
    }

parser = HfArgumentParser((Arguments))

if len(sys.argv) == 1:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_json_file(json_file=os.path.abspath("configs/distillation.json"))[0]
elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
else:
    args = parser.parse_args_into_dataclasses()[0]

set_seed(args.seed)

os.environ['WANDB_DISABLED'] = 'true'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv(args.train_csv)
test_df = pd.read_csv(args.test_csv)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

if args.dataset_name == 'HCT':
    class_text = ['plastic', 'fabric', 'metal' ,'steel', 'rubber']
    class_names = class_text
    num_labels = len(class_text)
    id2label = {0: 'plastic', 1: 'fabric', 2: 'metal', 3: 'steel', 4: 'rubber'}
    label2id = {label:id for id,label in id2label.items()}

elif args.dataset_name == 'SSVTP':

    class_names = ['canvas', 'cotton', 'jeans', 'marble', 'metal', 'nylon', 'plastic', 'polyster', 'towel'] 

    num_labels = len(class_names)
    id2label = {idx: label for idx, label in enumerate(class_names)}

    label2id = {label:id for id,label in id2label.items()}

elif args.dataset_name in ('Comine_HCT_SSVTP', 'Combine_HCT_SSVTP'):
    class_text = ['steel', 'plastic', 'fabric', 'metal', 'rubber', 'marble', 'towel', 'cotton', 'nylon', 'fleece', 'wood', 'silicon']
    class_names = class_text
    num_labels = len(class_text)
    id2label = {idx: label for idx, label in enumerate(class_text)}
    label2id = {label: idx for idx, label in id2label.items()}

else:
    raise ValueError(f"Unsupported dataset: {args.dataset_name}")


tokenizer = AutoTokenizer.from_pretrained(args.text_encoder, model_max_length= args.max_length, padding = "max_length", truncation=True)

image_processor = AutoImageProcessor.from_pretrained(args.image_encoder)

train_dataset = VLM_dataset(data_root= args.data_root,
                              df=train_df,
                              label2id = label2id,
                              tokenizer=tokenizer,
                              image_processor = image_processor,
                              VLM_disitillation=True
                            )

test_dataset = VLM_dataset(data_root= args.data_root,
                              df=test_df,
                              label2id = label2id,
                              tokenizer=tokenizer,
                              image_processor = image_processor,
                              VLM_disitillation=True
                            )

if args.pre_trained_vision_weights:
    vision_model = AutoModelForImageClassification.from_pretrained(args.image_encoder,num_labels=num_labels, 
                    id2label=id2label,
                    label2id=label2id,
                    ignore_mismatched_sizes=True) 
    
else: 
    vision_config = AutoConfig.from_pretrained(args.image_encoder,num_labels=num_labels, 
                    id2label=id2label,
                    label2id=label2id,)
    vision_model = AutoModelForImageClassification.from_config(vision_config)

if args.pre_trained_text_weights:
    text_model = AutoModelForSequenceClassification.from_pretrained(args.text_encoder,num_labels=num_labels, 
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True) 
else: 
    text_config = AutoConfig.from_pretrained(args.text_encoder,num_labels=num_labels, 
                id2label=id2label,
                label2id=label2id,)
    text_model = AutoModelForSequenceClassification.from_config(text_config)


training_args = DistillationTrainingArguments(
    args.output_dir,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    report_to="none",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    dataloader_num_workers=args.dataloader_num_workers,
    num_train_epochs=args.num_train_epochs,
    weight_decay=args.weight_decay,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    save_total_limit=2,
    metric_for_best_model="eval_loss",
    logging_dir='logs',
    alpha=args.alpha,
    beta=args.beta,
    temperature=args.temperature,
    distillation_type=args.distillation_type,
    hidden_rep_loss_weight= args.hidden_rep_loss_weight,
    ce_loss_weight= args.ce_loss_weight
)

trainer = DistillationTrainer(
    vision_model,
    training_args,
    teacher_model=text_model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator = collate_fn,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train(resume_from_checkpoint=args.resume_chk)

test_loader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

vision_model.to(device)
vision_model.eval()

text_model.to(device)
text_model.eval()

# -------------------------------------------------
# Extract embeddings
# -------------------------------------------------
tactile_feats, tactile_labels = [], []
lang_feats, lang_labels = [], []

with torch.no_grad():
    for batch in test_loader:

        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Vision student
        out_v = vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        v_cls = out_v.hidden_states[-1][:, 0, :]
        tactile_feats.append(v_cls.detach().cpu().numpy())
        tactile_labels.append(labels.detach().cpu().numpy())

        # Text teacher
        out_t = text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        try:
            t_cls = out_t.hidden_states[-1][:, 0, :]
        except AttributeError:
            t_cls = out_t.encoder_hidden_states[-1][:, 0, :]

        lang_feats.append(t_cls.detach().cpu().numpy())
        lang_labels.append(labels.detach().cpu().numpy())

tactile_feats = np.concatenate(tactile_feats, axis=0)
tactile_labels = np.concatenate(tactile_labels, axis=0)

lang_feats = np.concatenate(lang_feats, axis=0)
lang_labels = np.concatenate(lang_labels, axis=0)

print("Tactile:", tactile_feats.shape)
print("Language:", lang_feats.shape)
print("Label match rate:", (tactile_labels == lang_labels).mean())

# -------------------------------------------------
# Save separately
# -------------------------------------------------
np.savez(
    os.path.join(args.output_dir, "tactile_embeddings_raw.npz"),
    features=tactile_feats,
    labels=tactile_labels,
    class_names=np.array(class_names),
)

np.savez(
    os.path.join(args.output_dir, "language_embeddings_raw.npz"),
    features=lang_feats,
    labels=lang_labels,
    class_names=np.array(class_names),
)

print("Saved embeddings successfully.")

vision_model.save_pretrained(args.output_dir+"/best_weights/")
image_processor.save_pretrained(args.output_dir+"/best_weights/")
