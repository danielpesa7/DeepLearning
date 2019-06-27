#!/usr/bin/env python
# coding: utf-8


import torch
from torch import nn, optim
from torchvision import transforms, models
import matplotlib.pyplot as plt
import sys
from PIL import Image
from torchvision import transforms

imagen = input('¿Cuál es la imágen a predecir?: \n')
input_image = Image.open(imagen)

model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
print('Making the prediction...')
model.eval()

preprocess = transforms.Compose([transforms.Resize((700,900)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                      std=[0.229, 0.224, 0.225])])


input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) 

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

# create a color pallette, selecting a color for each class
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)

colors_category = r.getcolors()
colors_category = [colors_category[i][1] for i in range(len(colors_category))]
classes = ('__background__',  # always index 0
           'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


plt.figure(figsize=(12,8))  # create a plot figure
plt.suptitle('Semantic Segmentation')
# create the first of two panels and set current axis
plt.subplot(1, 2, 1) # (rows, columns, panel number)
plt.title('Original Image')
plt.imshow(input_image)

# create the second panel and set current axis
plt.subplot(1, 2, 2)
plt.title('Segmented Image')
plt.imshow(r)

adder = 0
for i in colors_category[1:]:
    plt.text(adder,input_image.size[1]+(input_image.size[1]/9),classes[i],bbox=dict(facecolor = 'gray', alpha = 0.6))
    adder += (input_image.size[0]/6)
plt.show()