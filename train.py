import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from data_loader import OCTDataset, WeightedSampler
from vit import ViT

image_size = 256
patch_size = int(256/16)
num_patches = int((image_size*image_size)/(patch_size*patch_size))
num_channels = 1
patch_dim = num_channels * patch_size * patch_size
num_classes = 4

# Finetune parameters
dim = 64
heads = 8
mlp_dim=128
depth = 8
dim_head = 64
dropout = 0.
emb_dropout = 0.
EPOCHS = 30

train_dataset_root = 'C:\\Major Project\\Dataset\\train'
weighted_sampler = WeightedSampler()
train_samples_weight, dataset_size = weighted_sampler.calculateWeights(train_dataset_root)
train_weighted_sampler = WeightedRandomSampler(train_samples_weight, dataset_size)
train_dataset = OCTDataset(train_dataset_root)
train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_weighted_sampler)

test_dataset_root = 'C:\\Major Project\\Dataset\\test'
test_dataset = OCTDataset(test_dataset_root)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

#img = torch.randn(1, 1, 256, 256)
#print(model)
#print('shape of output tensor: ', model(img).shape)    
#print(model.training)

# Training 
def train_epoch(model, optimizer, data_loader, loss_history):
    model.train()
    
    total_loss = 0
    total_samples = len(data_loader.dataset)
    
    for i, (data, target) in enumerate(data_loader):
        
        data = data.to(device)
        
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        
        loss = F.nll_loss(output, target)
        loss.backward()

        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / total_samples
    print("Train Loss: ", avg_loss)
    loss_history.append(avg_loss)

def evaluate(model, data_loader, loss_history, accuracy_history):
    model.eval()
    
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    accuracy = correct_samples / total_samples
    loss_history.append(avg_loss)
    accuracy_history.append(round(accuracy.item(), 2))
    
    print('\nTest Loss: ' + '{:.4f}'.format(avg_loss) +
          '     Accuracy: ' + '{:4.2f}'.format(100.0 * accuracy) + '%\n')
    
    return accuracy.item()

model = ViT(patch_size=patch_size, patch_dim=patch_dim, num_patches=num_patches, dim=dim, emb_dropout=emb_dropout, heads=heads, num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

train_loss_history, test_loss_history, test_accuracy_history = [], [], []

max_acc = sys.float_info.min
best_model_path = ''
for epoch in range(EPOCHS):
    print('Epoch:', epoch + 1)

    train_epoch(model, optimizer, train_loader, train_loss_history)
    acc = evaluate(model, test_loader, test_loss_history, test_accuracy_history)
    
    if (epoch + 1) % 5 == 0:
        torch.save(model, 'C:\\Major Project\\Saved Models\\m' + str(epoch+1) + '.pth')
        
    if acc > max_acc:
        
        try:
            os.remove(best_model_path)
        except:
            pass
        
        torch.save(model, 'C:\\Major Project\\Saved Models\\mbest' + str(epoch+1) + '.pth')
        best_model_path = 'C:\\Major Project\\Saved Models\\mbest' + str(epoch+1) + '.pth'
        max_acc = acc


plt.plot(np.array(train_loss_history), label='train loss')
plt.xticks(range(EPOCHS), range(1, EPOCHS+1))
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Train Loss')
plt.legend()
plt.savefig('C:\\Major Project\\Figures\\train_loss.png')
plt.close()

plt.plot(np.array(test_loss_history), label='test loss')
plt.xticks(range(EPOCHS), range(1, EPOCHS+1))
plt.title('Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Test Loss')
plt.legend()
plt.savefig('C:\\Major Project\\Figures\\test_loss.png')
plt.close()

plt.plot(np.array(test_accuracy_history), label='test accuracy')
plt.xticks(range(EPOCHS), range(1, EPOCHS+1))
plt.title('Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('C:\\Major Project\\Figures\\test_acc.png')
plt.close()