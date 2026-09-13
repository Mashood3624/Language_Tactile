import os
import pandas as pd
import sys
import numpy as np
import torch
from utilities.data import VLM_dataset, TAG_dataset, YCB_dataset, ICRA18_dataset, FEEL_dataset
from utilities.utils import Arguments
from transformers import AutoImageProcessor,AutoConfig, AutoModelForImageClassification, EarlyStoppingCallback
from transformers import TrainingArguments, Trainer, set_seed, HfArgumentParser
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def _freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

def collate_fn(examples):
    pixel_values = torch.cat([example["pixel_values"] for example in examples], dim=0)
    labels = torch.tensor([example["labels"] for example in examples])
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }

parser = HfArgumentParser((Arguments))

if len(sys.argv) == 1:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_json_file(json_file=os.path.abspath("configs/fewshot.json"))[0]
elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
else:
    args = parser.parse_args_into_dataclasses()[0]

if args.few_shot or args.k_shot is not None or args.train_percent is not None:
    raise ValueError("Provide a sampled train_csv for few-shot training; few_shot, k_shot, and train_percent are not implemented.")

set_seed(args.seed)

os.environ['WANDB_DISABLED'] = 'true'

if args.dataset_name == "HCT":
    id2label = {0: 'leather', 1: 'woven', 2: 'wall', 3: 'table', 4: 'carpet', 5: 'polyester', 6: 'wood', 7: 'aluminium', 8: 'paper', 9: 'cable', 10: 'synthetic', 11: 'towel', 12: 'stone'}
    class_names = ['leather', 'woven', 'wall', 'table', 'carpet', 'polyester', 'wood', 'aluminium', 'paper', 'cable', 'synthetic', 'towel', 'stone']
    label2id = {label:id for id,label in id2label.items()}

elif args.dataset_name == "SSVTP":
    class_names = ['canvas', 'cardboard', 'iron', 'jeans', 'polyster', 'steel', 'wool']

    id2label = {idx: label for idx, label in enumerate(class_names)}

    label2id = {label: idx for idx, label in id2label.items()}

elif args.dataset_name in ("Combine_HCT_SSVTP", "Comine_HCT_SSVTP"):
    class_names = ['leather', 'aluminium', 'towel', 'carpet', 'wood', 'polyester', 'table', 'paper', 'synthetic', 'wall', 'stone', 'cable', 'woven',
                    'jeans', 'wool', 'canvas', 'cardboard', 'polyster', 'iron', 'steel']

    id2label = {idx: label for idx, label in enumerate(class_names)}

    label2id = {label: idx for idx, label in id2label.items()}

elif args.dataset_name == "TAG":
    id2label = {0: 'Concrete', 1: 'Plastic', 2: 'Glass', 3: 'Wood', 4: 'Metal', 5: 'Brick', 6: 'Tile', 7: 'Leather', 8: 'Synthetic Fabric', 9: 'Natural Fabric',
                10: 'Ruber', 11: 'Paper', 12: 'Tree', 13: 'Grass', 14: 'Soil', 15: 'Rock', 16: 'Gravel', 17: 'Sand', 18: 'Plants', 19: 'Others'}
    class_names = ['Concrete' , 'Plastic' , 'Glass' , 'Wood' , 'Metal' , 'Brick' , 'Tile' , 'Leather' , 'Synthetic Fabric' , 'Natural Fabric' ,
                    'Ruber' , 'Paper' , 'Tree' , 'Grass' , 'Soil' , 'Rock' , 'Gravel' , 'Sand' , 'Plants' , 'Others']
    label2id = {label:id for id,label in id2label.items()}

elif args.dataset_name == "YCB":
    id2label = {0: '004_sugar_box', 1: '005_tomato_soup_can', 2: '006_mustard_bottle', 3: '021_bleach_cleanser', 4: '025_mug', 5: '035_power_drill',
                6: '037_scissors', 7: '042_adjustable_wrench', 8: '048_hammer', 9: '055_baseball'}
    class_names = ['004_sugar_box' , '005_tomato_soup_can' , '006_mustard_bottle' , '021_bleach_cleanser' , '025_mug' , '035_power_drill' ,
                    '037_scissors' , '042_adjustable_wrench' , '048_hammer' , '055_baseball']
    label2id = {label:id for id,label in id2label.items()}

elif args.dataset_name == "ICRA18":
    id2label = {0: 'cotton', 1: 'satin', 2: 'polyester', 3: 'denim', 4: 'garbardine', 5: 'broad cloth', 6: 'parka', 7: 'leather', 8: 'crepe', 9: 'corduroy', 10: 'velvet',
                 11: 'flannel', 12: 'fleece', 13: 'hairy', 14: 'wool', 15: 'knit', 16: 'net', 17: 'suit', 18: 'woven', 19: 'other'
                }
    class_names = ['cotton', 'satin', 'polyester', 'denim', 'garbardine', 'broad cloth', 'parka', 'leather', 'crepe', 'corduroy', 'velvet', 'flannel', 'fleece',
                    'hairy', 'wool', 'knit', 'net', 'suit', 'woven', 'other']
    label2id = {label:id for id,label in id2label.items()}

elif args.dataset_name == 'FEEL':
    id2label = {0: '3d_printed_blue_connector', 1: '3d_printed_blue_vase', 2: '3d_printed_white_ball', 3: 'aspirin', 4: 'baby_cup', 5: 'bandaid_box', 6: 'blue_painted_glass',
                7: 'brown_paper_cup_2_upside', 8: 'chocolate_shake', 9: 'cinnamon', 10: 'dog_toy_ice_cream_cone', 11: 'emergency_stop_button_for_sawyer', 12: 'fake_flower_in_pot',
                13: 'fox_head', 14: 'french_dip', 15: 'international_travel_adapter', 16: 'lime', 17: 'mentos_gum_can', 18: 'metal_cylinder_with_holes', 19: 'monofilament_line',
                20: 'moroccan_mint_tea_box', 21: 'muffin', 22: 'ogx_shampoo', 23: 'onion', 24: 'peanut_butter', 25: 'peppermint_altoids_box', 26: 'pig', 27: 'plastic_chicken',
                28: 'plastic_cow', 29: 'plastic_duck', 30: 'plastic_sheep', 31: 'plastic_watering_can', 32: 'plastic_whale', 33: 'playdoh_container', 34: 'ponds_dry_skin_cream',
                35: 'purple_small_plastic_fruit', 36: 'red_apple', 37: 'red_turtle', 38: 'rubics_cube', 39: 'set_small_plastic_men_blue_guy', 40: 'set_small_plastic_men_green_guy',
                41: 'set_small_plastic_men_police_man', 42: 'set_small_plastic_men_red_racer', 43: 'small_coffe_cup', 44: 'soda_can', 45: 'soft_beer_bottle_holder',
                46: 'soft_blue_hexagon', 47: 'soft_red_cube', 48: 'soft_zebra', 49: 'stuffed_beachball', 50: 'webcam_box'}
    class_names = id2label.values()
    label2id = {label:id for id,label in id2label.items()}

else:
    raise ValueError(f"Unknown dataset: {args.dataset_name}")

if args.pre_trained_vision_weights:
    model = AutoModelForImageClassification.from_pretrained(args.image_encoder,id2label=id2label,label2id=label2id,num_labels= len(id2label),ignore_mismatched_sizes=True,output_hidden_states=False)
else:
    config = AutoConfig.from_pretrained(args.image_processor_name or args.image_encoder,id2label=id2label,label2id=label2id,num_labels= len(id2label),output_hidden_states=False)
    model = AutoModelForImageClassification.from_config(config=config)

if args.freeze_vision:
    _freeze_params(model.base_model)

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total, trainable = count_params(model)

print(f"Total params: {total:,}")
print(f"Trainable params: {trainable:,}")
print(f"Trainable %: {100 * trainable/total:.4f}%")

image_processor = AutoImageProcessor.from_pretrained(args.image_processor_name or args.image_encoder)

aug = None
if args.augmentation:
    import albumentations as A

    aug = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.GaussNoise(std_range=(np.sqrt(10.0) / 255, np.sqrt(50.0) / 255), p=0.5),
        A.MotionBlur(p=0.2),
        A.RandomBrightnessContrast(p=0.5),
        A.CoarseDropout(num_holes_range=(4, 4), hole_height_range=(32, 32), hole_width_range=(32, 32), fill=0, p=0.5),
    ], seed=args.seed)

if args.dataset_name in ("HCT", "SSVTP", "Combine_HCT_SSVTP", "Comine_HCT_SSVTP"):

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_dataset = VLM_dataset(data_root= args.data_root,
                                df=train_df,
                                label2id = label2id,
                                image_processor = image_processor,
                                image_classification = True,
                                augmentations = aug
                                )

    test_dataset = VLM_dataset(data_root= args.data_root,
                                df=test_df,
                                label2id = label2id,
                                image_processor = image_processor,
                                image_classification = True
                                )

elif args.dataset_name == "TAG":
    train_dataset = TAG_dataset(
       root = args.data_root, image_processor = image_processor, mode='train')

    test_dataset = TAG_dataset(
       root = args.data_root, image_processor = image_processor, mode='test')

elif args.dataset_name == "YCB":
    train_df = pd.read_csv(args.train_csv)

    train_df, test_df = train_test_split(train_df, test_size=0.2)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_dataset = YCB_dataset(
       data_root = args.data_root, image_processor = image_processor, df = train_df, label2id=label2id)

    test_dataset = YCB_dataset(
       data_root = args.data_root, image_processor = image_processor, df = test_df, label2id=label2id)

elif args.dataset_name == "ICRA18":

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    train_dataset = ICRA18_dataset(
       data_root = args.data_root, image_processor = image_processor, df = train_df, label2id=label2id)

    test_dataset = ICRA18_dataset(
       data_root = args.data_root, image_processor = image_processor, df = test_df, label2id=label2id)

elif args.dataset_name == "FEEL":

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    train_dataset = FEEL_dataset(
       data_root = args.data_root, image_processor = image_processor, df = train_df, label2id=label2id)

    test_dataset = FEEL_dataset(
       data_root = args.data_root, image_processor = image_processor, df = test_df, label2id=label2id)

print(len(train_dataset))
print(len(test_dataset))
print(len(test_dataset)+len(train_dataset))

train_args = TrainingArguments(
    args.output_dir,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    num_train_epochs=args.num_train_epochs,
    weight_decay=args.weight_decay,
    load_best_model_at_end=True,
    save_total_limit=2,
    dataloader_num_workers=args.dataloader_num_workers,
    metric_for_best_model="eval_loss",
    logging_dir='logs',
    report_to="none",
)

trainer = Trainer(
    model,
    train_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator = collate_fn,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],

)

trainer.train(resume_from_checkpoint=args.resume_chk)

model.save_pretrained(os.path.join(args.output_dir, "best_weights"))
image_processor.save_pretrained(os.path.join(args.output_dir, "best_weights"))

output = trainer.predict(test_dataset)
predictions = np.argmax(output.predictions, axis=1)
labels = output.label_ids

accuracy = accuracy_score(labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)

with open(os.path.join(args.output_dir, "results" + os.path.basename(os.path.normpath(args.output_dir)) + ".txt"), "w") as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%" + "\n")
    f.write(f"Precision: {precision:.2f}" + "\n")
    f.write(f"Recall: {recall:.2f}" + "\n")
    f.write(f"F1-score: {f1:.2f}" + "\n")

cm = confusion_matrix(labels, predictions, labels=list(id2label))

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "confusion_mtrx.png"))
plt.close()
