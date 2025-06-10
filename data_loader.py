import glob
import numpy as np
import cv2
from collections import Counter
from pandas.core.common import flatten
import torch
from torch.utils.data import Dataset

#data transforms???

class OCTDataset(Dataset):
    
    def __init__(self, root):
        self.root = root
        
        self.image_paths = []
        for folder in glob.glob(self.root + '/*'):
            self.image_paths.append(glob.glob(folder + '/*'))
        
        self.image_paths = list(flatten(self.image_paths))
        
        classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        self.idx_to_class = {i:j for i, j in enumerate(classes)}
        self.class_to_idx = {value:key for key,value in self.idx_to_class.items()}
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        
        #normalize image
        #one hot encoding of label
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        image = np.array(image, dtype='float32')
        image /= 255.
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        
        label = image_filepath.split('\\')[-2]
        label = self.class_to_idx[label]
        
        return image, label

class WeightedSampler():
        
    def calculateWeights(self, root):
        class_list = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        idx_to_class = {i:j for i, j in enumerate(class_list)}
        class_to_idx = {value:key for key,value in idx_to_class.items()}
        
        image_classes = []
        for folder in glob.glob(root + '/*'):
            image_classes.append(glob.glob(folder + '/*'))
        
        image_classes = list(flatten(image_classes))
        image_classes = [image_class.split('\\')[-2] for image_class in image_classes]
        
        counter = Counter(image_classes)
        class_count = [i for i in counter.values()]
        class_weights = 1./torch.tensor(class_count, dtype=torch.float)
        
        train_samples_weight = [class_weights[class_to_idx[label]] for label in image_classes]
        dataset_size = len(train_samples_weight)
        
        return train_samples_weight, dataset_size
        
'''   
dataset_root = 'C:\\Major Project\\Dataset\\train'
weighted_sampler = WeightedSampler()
train_samples_weight, dataset_size = weighted_sampler.calculateWeights(dataset_root)
print(train_samples_weight, dataset_size)


dataset_root = 'C:\\Major Project\\Dataset\\train'
dataset = OCTDataset(dataset_root)

print(dataset[0][0].shape)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)


for i in enumerate(data_loader):
    print(i[1][1])
    break
'''