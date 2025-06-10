from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
import scikitplot as skplt
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader 
from data_loader import OCTDataset
from vit import ViT

test_dataset_root = 'C:\\Major Project\\Dataset\\test'
test_dataset = OCTDataset(test_dataset_root)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model = torch.load('C:\\Major Project\\Saved Models (lr = 0.0001)\\mbest26.pth')
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
idx_to_class = {i:j for i, j in enumerate(classes)}

total_samples = len(test_loader.dataset)
correct_samples = 0
total_loss = 0

y_pred = []
y_test = []

with torch.no_grad():
    for data, target in test_loader:
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target, reduction='sum')
        _, pred = torch.max(output, dim=1)
        
        y_pred.append(idx_to_class[pred.item()])
        y_test.append(idx_to_class[target.item()])
        
        total_loss += loss.item()
        correct_samples += pred.eq(target).sum()

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average=None)
recall = metrics.recall_score(y_test, y_pred, average=None)
f1_score = metrics.f1_score(y_test, y_pred, average=None)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)

# Calculate classification report
classification_report = metrics.classification_report(y_test, y_pred)
print("Classification report:\n", classification_report)

skplt.metrics.plot_confusion_matrix(y_test, y_pred, figsize=(12,12))
plt.title('Confusion Matrix')
plt.show()