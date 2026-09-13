import os
import cv2
from torch.utils.data import Dataset

class VLM_dataset(Dataset):
    def __init__(self, data_root, df, label2id, image_processor, tokenizer=None, image_classification=False, VLM_classification=False, class_text=None,VLM_disitillation=False, augmentations= None):
        self.data_root = data_root
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_classification = image_classification
        self.VLM_classification = VLM_classification
        self.label2id = label2id
        self.class_text = class_text
        self.VLM_disitillation = VLM_disitillation
        self.augmentations = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root, self.df["tactile"][idx])
        class_label = self.label2id[self.df["class"][idx]]

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        if self.VLM_classification:
            inputs = self.tokenizer(text=self.class_text,padding = "max_length",truncation=True,return_tensors="pt")
            inputs["pixel_values"]  = self.image_processor(image, return_tensors="pt").pixel_values
            inputs["labels"] = class_label
        elif self.VLM_disitillation:

            inputs = self.tokenizer(text=self.df["caption"][idx],padding = "max_length",truncation=True)
            inputs["pixel_values"]  = self.image_processor(image, return_tensors="pt").pixel_values
            inputs["labels"] = class_label

        elif self.image_classification:

            inputs = self.image_processor(image, return_tensors="pt")
            inputs["labels"] = class_label
        else:
            inputs = self.tokenizer(text=self.df["caption"][idx],padding = "max_length",truncation=True)
            inputs["pixel_values"] = self.image_processor(image, return_tensors="pt").pixel_values
            inputs["labels"] = class_label

        return inputs

class SSVTP_dataset(Dataset):
    def __init__(self, data_root, df, label2id, image_processor, tokenizer=None, image_classification=False, VLM_classification=False, class_text=None,VLM_disitillation=False):
        self.data_root = data_root
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_classification = image_classification
        self.VLM_classification = VLM_classification
        self.label2id = label2id
        self.class_text = class_text
        self.VLM_disitillation = VLM_disitillation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root, self.df["tactile"][idx])
        class_label = self.label2id[self.df["class"][idx]]

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.VLM_classification:
            inputs = self.tokenizer(text=self.class_text,padding = "max_length",truncation=True,return_tensors="pt")
            inputs["pixel_values"]  = self.image_processor(image, return_tensors="pt").pixel_values
            inputs["labels"] = class_label
        elif self.VLM_disitillation:

            inputs = self.tokenizer(text=self.df["caption"][idx],padding = "max_length",truncation=True)
            inputs["pixel_values"]  = self.image_processor(image, return_tensors="pt").pixel_values
            inputs["labels"] = class_label

        elif self.image_classification:
            inputs = self.image_processor(image, return_tensors="pt")
            inputs["labels"] = class_label
        else:
            inputs = self.tokenizer(text=self.df["caption"][idx],padding = "max_length",truncation=True)
            inputs["pixel_values"] = self.image_processor(image, return_tensors="pt").pixel_values
        return inputs

class TAG_dataset(Dataset):
    def __init__(
        self,
        root,
        image_processor,
        transform=None,
        target_transform=None,
        mode="train",
        label="full",
    ):
        self.dataroot = os.path.join(root, "dataset")
        self.image_processor = image_processor
        self.transform = transform
        self.target_transform = target_transform
        self.label = label

        # ---- select index file ----
        if mode == "train":
            index_file = os.path.join(root, "train.txt")
        elif mode == "test":
            index_file = os.path.join(root, "test.txt")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ---- read and precompute everything ----
        self.samples = []

        with open(index_file, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        for line in lines:
            raw, target_str = line.split(",")
            target = int(target_str)

            # label remapping
            if self.label == "hard":
                target = 1 if target in (7, 8, 9, 11, 13) else 0

            idx = os.path.basename(raw)
            img_dir = os.path.join(self.dataroot, raw[:16], "gelsight_frame")
            image_path = os.path.join(img_dir, idx)

            self.samples.append((image_path, target))

        self.length = len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        # load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        inputs = self.image_processor(image, return_tensors="pt")

        if self.transform is not None:
            inputs = self.transform(inputs)

        inputs["labels"] = target
        return inputs

    def __len__(self):
        return self.length

class YCB_dataset(Dataset):

    def __init__(self, data_root, image_processor, df, label2id):

        self.data_root = data_root
        self.df = df.reset_index(drop=True)
        self.image_processor = image_processor
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root, self.df["Path"][idx])
        class_label = self.label2id[self.df["Object"][idx]]

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        inputs = self.image_processor(image, return_tensors="pt")
        inputs["labels"] = class_label

        return inputs

class ICRA18_dataset(Dataset):

    def __init__(self, data_root, image_processor, df, label2id):

        self.data_root = data_root
        self.df = df.reset_index(drop=True)
        self.image_processor = image_processor
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root, self.df["path"][idx])
        class_label = self.df["Textile_type"][idx]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        inputs = self.image_processor(image, return_tensors="pt")
        inputs["labels"] = self.label2id[class_label]

        return inputs

class FEEL_dataset(Dataset):

    def __init__(self, data_root, image_processor, df, label2id):

        self.data_root = data_root
        self.df = df.reset_index(drop=True)
        self.image_processor = image_processor
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_root, self.df["images_path"][idx])
        class_label = self.label2id[self.df["obj_names"][idx]]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        inputs = self.image_processor(image, return_tensors="pt")
        inputs["labels"] = class_label

        return inputs
