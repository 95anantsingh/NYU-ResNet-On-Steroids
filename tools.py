
import torch
import numpy as np
import matplotlib.pyplot as plt

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def plotImg(dataset,rng:int,name:str, transformed=False):
    fig = plt.figure(figsize=(12,3))
    if not transformed:
        images = np.copy(dataset.data)[0:rng]
        labels = np.copy(dataset.targets)[0:rng]
    for index in range(rng):
        plt.subplot(1,5,index+1)
        if transformed:
            image, label = dataset[index]
            image = image.numpy()
            plt.imshow(np.transpose(image, (2,1,0)))
        else:
            image, label = images[index], labels[index]
            plt.imshow(image)
        ax = plt.gca()
        ax.set_title(classes[label])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.suptitle(name)
    plt.savefig(name+".png")
    