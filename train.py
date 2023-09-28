from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights, FastRCNNPredictor
from dataset import VOCDataset
from pprint import pprint
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, RandomAffine, Resize, ColorJitter

def collate_fn(batch):
    all_images = []
    all_labels = []
    for image, label in batch:
        all_images.append(image)
        all_labels.append(label)
    return all_images, all_labels

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = Compose([
        # RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        Resize((416, 416)),
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.255)),
    ])

    val_transform = Compose([
        Resize((416, 416)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.255)),
    ])

    train_set = VOCDataset(root='./data/voc', year='2007', image_set='train', download=False, transform = train_transform)
    val_set = VOCDataset(root='./data/voc', year='2007', image_set='val', download=False, transform = val_transform)

    train_params = {
        'batch_size': 2,
        'shuffle': True,
        'num_workers': 1,
        'drop_last': True,
        'collate_fn': collate_fn, # collate_fn is used to merge the list of samples to form a mini-batch.
        }
    
    val_params = {
        'batch_size': 2,
        'shuffle': False,
        'num_workers': 1,
        'drop_last': False,
        'collate_fn': collate_fn, # collate_fn is used to merge the list of samples to form a mini-batch.
    }

    train_dataloader = DataLoader(train_set, **train_params)
    val_dataloader = DataLoader(val_set, **val_params)

    model = fasterrcnn_resnet50_fpn(weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=model.roi_heads.box_predictor.cls_score.in_features, num_classes=21)
    model.to(device)
    model.train()
    for images, targets in train_dataloader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_components = model(images, targets)
        losses = sum(loss for loss in loss_components.values())

if __name__ == '__main__':
    train()     