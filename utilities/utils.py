from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class Arguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    output_dir: str = field(
        metadata={"help": "Path to save the outputs"},
    )
    
    image_encoder: str = field(
        metadata={"help": "Model name or path of vision encoder"},
    )
    
    text_encoder: str = field(
        metadata={"help": "Model name or path of text encoder"},
    )

    dataset_name: str = field(
        metadata={"help": "dataset name eg. HCT, TAG, SSVTP, FEEL, YCB, or ICRA18"},
    )
    
    data_root: str = field(
        metadata={"help": "Path to dataset"},
    )
    
    train_csv: str = field(
        metadata={"help": "Path for train dataset CSV file"},
    )
    
    test_csv: str = field(
        metadata={"help": "Path for test dataset CSV file"},
    )
    
    distillation_type: Optional[str] = field(
        default=None, metadata={"help": "Type of Distillation eg: Cosine_minimization, KD_logits"}
    )

    VLM_weights: Optional[str] = field(
        default=None, metadata={"help": "Path to VLM weights"}
    )
    
    image_processor_name: Optional[str] = field(
        default=None, metadata={"help": "Model name or path of image processor"}
    )
    
    seed: Optional[int] = field(
        default=1234, metadata={"help": "Batch size at training phase"}
    )
    
    pre_trained_vision_weights: Optional[bool] = field(
        default=True, metadata={"help": "Argument to load pre_trained weights of vision model"}
    )
    
    pre_trained_text_weights: Optional[bool] = field(
        default=True, metadata={"help": "Argument to load pre_trained weights of text model"}
    )
    
    pre_trained_VLM_weights: Optional[bool] = field(
        default=False, metadata={"help": "Argument to load pre_trained weights of text model"}
    )
    
    freeze_vision: Optional[bool] = field(
        default=False, metadata={"help": "Batch size at training phase"}
    )
    
    freeze_text: Optional[bool] = field(
        default=False, metadata={"help": "Batch size at training phase"}
    )
    
    per_device_train_batch_size: Optional[int] = field(
        default=16, metadata={"help": "Batch size at training phase"}
    )
    
    per_device_eval_batch_size: Optional[int] = field(
        default=16, metadata={"help": "Batch size at evaluation phase"}
    )

    dataloader_num_workers: Optional[int] = field(
        default=0, metadata={"help": "Number of workers for data loading"}
    )
    
    num_train_epochs: Optional[int] = field(
        default=50, metadata={"help": "Total number of epochs"}
    )
    
    weight_decay: Optional[float] = field(
        default=0.01, metadata={"help": "Weight decay rate for training the model"}
    )
    
    learning_rate: Optional[float] = field(
        default=2e-5, metadata={"help": "Learning rate for training the model"}
    )
    
    resume_chk: Optional[bool] = field(
        default=False, metadata={"help": "Resume training from last checkpoint"}
    )
    
    max_length: Optional[int] = field(
        default=128 , metadata={"help": "Resumce training from last checkpoint"}
    )
    # distilation parameters
    alpha: Optional[float] = field(
        default=0.5, metadata={"help": "KD_Logits Distilation parameter"}
    )
    
    temperature: Optional[float] = field(
        default=4.0 , metadata={"help": "KD_Logits Distilation parameter"}
    )

    hidden_rep_loss_weight: Optional[float] = field(
        default=0.25 , metadata={"help": "Cosine_minimization Distilation parameter"}
    )

    ce_loss_weight: Optional[float] = field(
        default=0.75 , metadata={"help": "Cosine_minimization Distilation parameter"}
    )

    beta: Optional[float] = field(
        default=8.0 , metadata={"help": "KD_Logits Distilation parameter"}
    )

    augmentation: Optional[bool] = field(
        default=False , metadata={"help": "Image Augmentation "}
    )

    train_percent: Optional[float] = field(
        default=None , metadata={"help": "Percentage of Training data"}
    )
    
    k_shot: Optional[int] = field(
        default=None , metadata={"help": "Percentage of Training data"}
    )
    few_shot: Optional[bool] = field(
        default=None , metadata={"help": "Percentage of Training data"}
    )
    

######################## https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/DKD.py

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

###################################### https://github.com/philschmid/knowledge-distillation-transformers-pytorch-sagemaker/blob/master/knowledge-distillation.ipynb

class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, beta= 8.0 ,hidden_rep_loss_weight=0.5, ce_loss_weight=0.5, distillation_type="KD_logits",  **kwargs):
        super().__init__(*args, **kwargs)
        
        self.distillation_type = distillation_type

        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        
        self.hidden_rep_loss_weight = hidden_rep_loss_weight
        self.ce_loss_weight = ce_loss_weight

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # place teacher on same device as student
        self._move_model_to_device(self.teacher,self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):

        # compute student output
        outputs_student = model(pixel_values = inputs["pixel_values"],labels=inputs["labels"],output_hidden_states=True)

        # compute teacher output
        with torch.no_grad():
          outputs_teacher = self.teacher(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],labels=inputs["labels"],output_hidden_states=True)
        if self.args.distillation_type == "KD_logits": 
            student_loss=outputs_student.loss

            # assert size
            assert outputs_student.logits.size() == outputs_teacher.logits.size()

            # Soften probabilities and compute distillation loss
            loss_function = nn.KLDivLoss(reduction="batchmean")
            loss_logits = (loss_function(
                F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1)) * (self.args.temperature ** 2))
            # Return weighted student loss
            loss = self.args.alpha * student_loss + (1. - self.args.alpha) * loss_logits

        elif self.args.distillation_type == "Cosine_minimization":
 
            try:
                teacher_hidden_representation = outputs_teacher.hidden_states[-1][:,0,:]
            except AttributeError:
                teacher_hidden_representation = outputs_teacher.encoder_hidden_states[-1][:,0,:]

            student_hidden_representation = outputs_student.hidden_states[-1][:,0,:]

            # assert size
            assert teacher_hidden_representation.size() == student_hidden_representation.size()

            ce_loss = nn.CrossEntropyLoss()
            cosine_loss = nn.CosineEmbeddingLoss()

            hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(inputs["labels"].size(0)).to(self.model.device))

            # Calculate the true label loss
            label_loss = ce_loss(outputs_student.logits, inputs["labels"])

            # Weighted sum of the two losses
            loss = self.args.hidden_rep_loss_weight * hidden_rep_loss + self.args.ce_loss_weight * label_loss

        elif self.args.distillation_type == "KD_feature":
            student_loss=outputs_student.loss

            try:
                teacher_hidden_representation = outputs_teacher.hidden_states[-1][:,0,:]
            except AttributeError:
                teacher_hidden_representation = outputs_teacher.encoder_hidden_states[-1][:,0,:]
            
            student_hidden_representation = outputs_student.hidden_states[-1][:,0,:]

            # assert size
            assert teacher_hidden_representation.size() == student_hidden_representation.size()

            loss_function = nn.KLDivLoss(reduction="batchmean")

            loss = loss_function(
                F.log_softmax(student_hidden_representation / self.args.temperature, dim=-1),
                F.softmax(teacher_hidden_representation / self.args.temperature, dim=-1))

            # Return weighted student loss
            loss = self.args.alpha * student_loss + (1. - self.args.alpha) * loss

        elif self.args.distillation_type == "Decouple_KD":
            student_loss=outputs_student.loss

            # assert size
            assert outputs_student.logits.size() == outputs_teacher.logits.size()

            loss = dkd_loss(
                outputs_student.logits,
                outputs_teacher.logits,
                inputs["labels"],
                self.args.alpha,
                self.args.beta,
                self.args.temperature,
            )

        else:
            raise ValueError(f"Unsupported distillation type: {self.args.distillation_type}")

        return (loss, outputs_student) if return_outputs else loss
