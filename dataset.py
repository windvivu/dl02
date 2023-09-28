#%%
from torchvision.datasets import VOCDetection
from pprint import pprint
import cv2
import numpy as np  
from torchvision.transforms import Compose, ToTensor, Normalize, RandomAffine, Resize, ColorJitter
import torch
# %%

class VOCDataset(VOCDetection):
    def __init__(self, root: str, year: str = "2007", image_set: str = "train", download: bool = False, transform = None):
        super().__init__(root, year, image_set, download, transform)
        self.categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, index: int):
        image, targets = super().__getitem__(index)
        old_h = int(targets['annotation']['size']['height'])
        old_w = int(targets['annotation']['size']['width'])
        _, new_h, new_w = image.shape

        labels = []
        bboxes = []
        output = {}
        for target in targets['annotation']['object']:
            label = self.categories.index(target['name'])
            labels.append(label)
            bbox = target['bndbox']
            xmin = int(float(bbox['xmin'])/old_w*new_w)
            ymin = int(float(bbox['ymin'])/old_h*new_h)
            xmax = int(float(bbox['xmax'])/old_w*new_w)
            ymax = int(float(bbox['ymax'])/old_h*new_h)
            bboxes.append([xmin, ymin, xmax, ymax])
      
        output['boxes'] = torch.FloatTensor(bboxes)
        output['labels'] = torch.LongTensor(labels)

        return image, output

#%%    
if __name__ == '__main__':
    train_transform = Compose([
        # RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        Resize((416, 416)),
        # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.255)),
    ])
    dataset = VOCDataset(root='./data/voc', year='2007', image_set='trainval', download=True, transform=train_transform)
    index = 2289 
    image, bboxes = dataset[index]

    print(image.shape)

    # image = np.array(image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

    # for bbox in bboxes:
    #     xmin, ymin, xmax, ymax = bbox
    #     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    
    # cv2.imwrite("test.jpg", image)
        
# %%
