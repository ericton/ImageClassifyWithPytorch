import torch
from collections import OrderedDict

from torch import nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    trans = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])

    img = trans(image).numpy()
    return img


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, topk=5, device=torch.device('cpu')):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    model.eval()
    image = process_image(Image.open(image_path))
    image = torch.from_numpy(image)
    model.to(device)
    image = image.to(device)
    log_ps = model.forward(image.view(1, *image.shape))
    ps = torch.exp(log_ps)
    top_p, top_c = ps.topk(topk, dim=1)
    c = top_c.view(-1).cpu().data.numpy()
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [index_to_class[each] for each in c]
    return top_p.view(-1).data.cpu().numpy(), top_classes


def getInNumber(model):
    if isinstance(model, nn.Sequential):
        return model[0].in_features
    elif isinstance(model, nn.Linear):
        return model.in_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a model to classify flowers',
    )
    parser.add_argument('input', action="store", type=str, help='flower to predict')
    parser.add_argument('model_dir', action="store", type=str)
    parser.add_argument('--top_k', action='store', type=int, default=5)
    parser.add_argument('--category_names', action='store', type=str, default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', default=False)
    arg = parser.parse_args()
    device = torch.device('cuda' if arg.gpu and torch.cuda.is_available() else 'cpu')
    cp = torch.load(arg.model_dir, map_location='cpu')
    model = getattr(models, cp['arch'], models.densenet121)()
    model.classifier = cp['classifier']
    model.load_state_dict(cp['state_dict'])
    model.class_to_idx = cp['class_to_idx']
    with open(arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    filepath = arg.input
    image = process_image(Image.open(filepath))
    p, c = predict(filepath, model, arg.top_k, device)
    labels = []
    for index in c:
        if str(index) in cat_to_name:
            labels.append(cat_to_name[str(index)])
        else:
            labels.append(index)
    print(p)
    print(labels)
    try:
        ax1 = plt.subplot(2, 1, 1)
        image = image.transpose((1, 2, 0))

        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean

        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
        plt.title(labels[0])
        plt.axis('off')
        plt.imshow(image)
        ax2 = plt.subplot(2, 1, 2)

        plt.gca().invert_yaxis()
        plt.barh(range(len(p)), p, tick_label=labels)
        plt.show()
    except Exception as e:
        print(e)
