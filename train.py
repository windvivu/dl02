from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights, FastRCNNPredictor, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from dataset import VOCDataset
from pprint import pprint
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, RandomAffine, Resize, ColorJitter
import torch.optim as optim
import argparse
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import numpy as np

def collate_fn(batch):
    all_images = []
    all_labels = []
    for image, label in batch:
        all_images.append(image)
        all_labels.append(label)
    return all_images, all_labels

def get_args():
    parser = argparse.ArgumentParser(description="Object detection and classifier")
    parser.add_argument("--data_path", type=str, default="data/voc", help="the root folder of the data")
    parser.add_argument("--epochs", default=50, type=int, help="Total number of epochs")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--image_size", default=416, type=int)
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=0,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    # parser.add_argument("--checkpoint", type=str, default="trained_models/last.pt", help="path to model checkpoint file")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to model checkpoint file")
    parser.add_argument("--log_path", type=str, default="tensorboard/pascal_voc")
    parser.add_argument("--save_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = Compose([
        # RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        Resize((args.image_size, args.image_size)),
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.255)),
    ])

    val_transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.255)),
    ])

    train_set = VOCDataset(root='./data/voc', year='2007', image_set='train', download=False, transform = train_transform)
    val_set = VOCDataset(root='./data/voc', year='2007', image_set='val', download=False, transform = val_transform)

    train_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 6,
        'drop_last': True,
        'collate_fn': collate_fn, # collate_fn is used to merge the list of samples to form a mini-batch.
        }
    
    val_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 6,
        'drop_last': False,
        'collate_fn': collate_fn, # collate_fn is used to merge the list of samples to form a mini-batch.
    }

    train_dataloader = DataLoader(train_set, **train_params)
    val_dataloader = DataLoader(val_set, **val_params)

    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=model.roi_heads.box_predictor.cls_score.in_features, num_classes=21)
    model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    writer = SummaryWriter(args.log_path)

    for epoch in range(args.epochs):
        
        model.train()
        train_loss = []
        progressbar = tqdm(train_dataloader, colour='cyan', ncols=100)

        for images, targets in progressbar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_components = model(images, targets)
            losses = sum(loss for loss in loss_components.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            train_loss.append(losses.item())
            avg_loss = np.mean(train_loss)
            progressbar.set_description("Epoch {}/{}. Loss {:0.4f}".format(epoch+1, args.epochs, avg_loss))
            writer.add_scalar("Train/Loss", avg_loss, epoch * len(train_dataloader) + iter)

if __name__ == '__main__':
    args = get_args()
    train(args)     