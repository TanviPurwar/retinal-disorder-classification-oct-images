import random
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from vit import ViT

images = []
images.append(glob.glob('C:\\Major Project\\Dataset\\test\\CNV\\*')[random.randint(0, 249)])
images.append(glob.glob('C:\\Major Project\\Dataset\\test\\DME\\*')[random.randint(0, 249)])
images.append(glob.glob('C:\\Major Project\\Dataset\\test\\DRUSEN\\*')[random.randint(0, 249)])
images.append(glob.glob('C:\\Major Project\\Dataset\\test\\NORMAL\\*')[random.randint(0, 249)])

model_path = 'C:\\Major Project\\Saved Models (lr = 0.0001)\\mbest26.pth'

model = torch.load(model_path)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
idx_to_class = {i:j for i, j in enumerate(classes)}

predictions = []
img = []
for image_filepath in images:
    i = cv2.imread(image_filepath)
    image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
    img.append(i)
    image = cv2.resize(image, (256, 256))
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image)

    output = F.log_softmax(model(image), dim=1)
    pred = torch.max(output, dim=1)
    idx = pred.indices[0].item()
    predictions.append(idx_to_class[idx])

f = plt.figure()
f.add_subplot(2, 2, 1)
plt.imshow(img[0])
plt.axis('off')
plt.title('PREDICTED LABEL: ' + predictions[0] + '\nGROUND TRUTH: ' + classes[0])

f.add_subplot(2, 2, 2)
plt.imshow(img[1])
plt.axis('off')
plt.title('PREDICTED LABEL: ' + predictions[1] + '\nGROUND TRUTH: ' + classes[1])

f.add_subplot(2, 2, 3)
plt.imshow(img[2])
plt.axis('off')
plt.title('PREDICTED LABEL: ' + predictions[2] + '\nGROUND TRUTH: ' + classes[2])

f.add_subplot(2, 2, 4)
plt.imshow(img[3])
plt.axis('off')
plt.title('PREDICTED LABEL: ' + predictions[3] + '\nGROUND TRUTH: ' + classes[3])

f.tight_layout(pad=1.0)
plt.show()
'''
  
ax[0,0].imshow(img[0])
ax[0,1].imshow(img[1])
ax[1,0].imshow(img[2])
ax[1,1].imshow(img[3])

ax[0,0].set_title('PREDICTED LABEL: ' + predictions[0] + '\nGROUND TRUTH: ' + classes[0])
ax[0,0].set_title('PREDICTED LABEL: ' + predictions[1] + '\nGROUND TRUTH: ' + classes[1])
ax[0,0].set_title('PREDICTED LABEL: ' + predictions[2] + '\nGROUND TRUTH: ' + classes[2])
ax[0,0].set_title('PREDICTED LABEL: ' + predictions[3] + '\nGROUND TRUTH: ' + classes[3])
plt.show()
'''