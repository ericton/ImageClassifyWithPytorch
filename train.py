import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import time
import argparse
from collections import OrderedDict

arg = None


def load_data(data_dir):
    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
        'valid': transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
        'test': transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])}

    image_datasets = {x: datasets.ImageFolder(data_dir + '/' + x,
                                              data_transforms[x])
                      for x in ['train', 'valid', 'test']}

    return image_datasets


def trainModel(datasets, model, epochs=10, lr=0.003, device=torch.device('cpu'), save_dir='checkpoint.pth'):
    map = {'train': True, 'valid': False, 'test': False}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=32,
                                                  shuffle=map[x])
                   for x in ['train', 'valid', 'test']}

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    model.to(device)
    cur_epoch = 0
    for e in range(epochs):

        train_start = time.time()
        running_loss = 0
        step = 0

        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            log_ps = model(images)

            loss = criterion(log_ps, labels)

            loss.backward()
            running_loss += loss.item()

            optimizer.step()
            cur_epoch = e
            step += 1
            if (step % 10 == 0):
                print(f'train step:{step}/{len(dataloaders["train"])}')
        model.cpu()
        checkpoint = {'epoch': cur_epoch,
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'class_to_idx': datasets['train'].class_to_idx,
                      'arch': arg.arch,
                      'classifier':model.classifier}
        torch.save(checkpoint, save_dir)
        model.to(device)
        print(f'epoch:{e+1},train time:{time.time()-train_start}')
        test_start = time.time()
        test_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            step = 0
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                log_ps = model.forward(inputs)
                loss = criterion(log_ps, labels)
                test_loss += loss.item()
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                step += 1
                if (step % 10 == 0):
                    print(f'valid step:{step}/{len(dataloaders["valid"])}')
            print(f'epoch:{e+1},test time:{time.time()-test_start}')
            print(f"train_loss:{running_loss/len(dataloaders['train'])}\n"
                  f"test_loss:{test_loss/len(dataloaders['test'])}\n"
                  f"accuracy:{accuracy/len(dataloaders['valid']):.3f}")
            model.train()
    return model


def getInNumber(model):
    if isinstance(model, nn.Sequential):
        return model[0].in_features
    elif isinstance(model, nn.Linear):
        return model.in_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a model to classify flowers',
    )
    parser.add_argument('data_dir', action="store", type=str, help='Data of typed flowers')
    parser.add_argument('--save_dir', action="store", type=str, default='checkpoint.pth')
    parser.add_argument('--learning_rate', action='store', type=float, default=0.002)
    parser.add_argument('--epochs', action='store', type=int, default=10)
    parser.add_argument('--hidden_unit', action='store', type=int, default=256)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--arch', action='store', default='densenet121', help='choose model')
    arg = parser.parse_args()
    device = torch.device('cuda' if arg.gpu and torch.cuda.is_available() else 'cpu')

    model = getattr(models, arg.arch, models.densenet121)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    myClassifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(getInNumber(model.classifier), arg.hidden_unit)),
                                              ('relu', nn.ReLU()),
                                              ('dropout', nn.Dropout(p=0.2)),
                                              ('fc2', nn.Linear(arg.hidden_unit, 102)),
                                              ('output', nn.LogSoftmax(dim=1)),
                                              ]))
    model.classifier = myClassifier
    datasets = load_data(arg.data_dir)
    trainModel(datasets, model, arg.epochs, arg.learning_rate, device, arg.save_dir)
