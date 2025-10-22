import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from tqdm.auto import tqdm

# This script is adapted from the MaskFormer tutorial available at
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/MaskFormer/Fine-tuning/Fine_tuning_MaskFormer_on_a_panoptic_dataset.ipynb


# Set float32 matmul precision
torch.set_float32_matmul_precision('medium')

TRAIN_PATH = "../data/train"

# Extract instance masks from annotated image
def extract_instance_masks(annotated_image):
    # Convert to numpy array
    annotated_array = np.array(annotated_image)
    
    # Extract masks with brightness above 128
    bright_mask = annotated_array.max(axis=2) > 128
    
    # Convert to HSV for unique hue values
    hsv_image = cv2.cvtColor(annotated_array, cv2.COLOR_RGB2HSV)
    
    # Get unique hue values
    unique_hues = np.unique(hsv_image[:,:,0][bright_mask])
    
    # Create instance masks
    instance_masks = []
    for hue in unique_hues:
        mask = np.all((hsv_image[:,:,0] == hue, bright_mask), axis=0)
        instance_masks.append(mask)
    
    return instance_masks


class CustomDataset(Dataset):
    def __init__(self, train_path, processor, transform=None):
        self.train_path = train_path
        self.processor = processor
        self.transform = transform
        self.annotated_files = [f for f in os.listdir(train_path) if f.endswith("_annotated.png")]
        
    def __len__(self):
        return len(self.annotated_files)
    
    def __getitem__(self, idx):
        annotated_file = self.annotated_files[idx]
        image_file = annotated_file.replace("_annotated.png", ".png")
        
        # Load image and annotated image
        image = np.array(Image.open(os.path.join(self.train_path, image_file)))
        annotated_image = Image.open(os.path.join(self.train_path, annotated_file))

        # Extract instance masks
        instance_masks = extract_instance_masks(annotated_image)
        
        # Create panoptic segmentation map, initializing all pixels to 0 (background)
        panoptic_map = np.zeros(annotated_image.size[::-1], dtype=np.int32)

        # Assign unique IDs to each instance, starting from 1
        for i, mask in enumerate(instance_masks):
            panoptic_map[mask] = i + 1
        
        # apply transforms (need to be applied on RGB values)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=panoptic_map)
            image, panoptic_map = transformed['image'], transformed['mask']
            # convert to C, H, W
            image = image.transpose(2,0,1)

        # Create instance_id_to_semantic_id mapping
        # 0 is background, 1 is the class for all instances (e.g., "duck")
        inst2class = {0: 0}  # background
        for i in range(len(instance_masks)):
            inst2class[i + 1] = 1  # all instances are class 1
        
        # Prepare inputs for the model
        inputs = self.processor([image], [panoptic_map], instance_id_to_semantic_id=inst2class, return_tensors="pt")
        
        # Remove the batch dimension
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}
        
        return inputs

# Initialise the processor and dataset
processor = Mask2FormerImageProcessor(ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)

# Image normalisation constants
DUCK_MEAN = np.array([0.36055567, 0.26455822, 0.1505872])
DUCK_STD = np.array([0.13891927, 0.10404531, 0.09613165])

image_transform = A.Compose([
    A.Resize(width=512, height=512),
    A.Normalize(mean=DUCK_MEAN.tolist(), std=DUCK_STD.tolist()),
])

# Training dataset
train_dataset = CustomDataset(TRAIN_PATH, processor, transform=image_transform)

# define custom collate function that defines how to batch examples together
def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels}

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)


# Mapping from class ID to description
id2label = {0: "background", 1: "duck"}

# Replace the segmentation head of the pre-trained model
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic",
                                                          id2label=id2label,
                                                          ignore_mismatched_sizes=True)


# Check for CUDA, then MPS, otherwise default to CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

running_loss = 0.0
num_samples = 0
for epoch in range(100):
    print("Epoch:", epoch)
    model.train()
    for idx, batch in enumerate(tqdm(train_dataloader)):
        # Reset the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            pixel_values=batch["pixel_values"].to(device),
            mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
            class_labels=[labels.to(device) for labels in batch["class_labels"]],
        )

        # Backward propagation
        loss = outputs.loss
        loss.backward()

        batch_size = batch["pixel_values"].size(0)
        running_loss += loss.item()
        num_samples += batch_size

        if idx % 100 == 0:
            print("Loss:", running_loss/num_samples)

        # Optimization
        optimizer.step()


# Save for later use in Python
model.save_pretrained("./model")
processor.save_pretrained("./processor")

