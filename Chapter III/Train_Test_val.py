import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

def tb_writer():
    timestr = time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_path + timestr)
    return writer

def misclassified_images(pred, writer, target, images, output, epoch, count=10):
    misclassified = (pred != target.data)
    for index, image_tensor in enumerate(images[misclassified][:count]):
        img_name = 'Epoch:{}-->Predict:{}-->Actual:{}'.format(epoch, LABEL[pred[misclassified].tolist()[index]],
                                                              LABEL[target.data[misclassified].tolist()[index]])
        writer.add_image(img_name, image_tensor, epoch)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.BN1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5)
        self.BN2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 64, 3)
        self.BN3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2)
        self.BN4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 3)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(self.BN1(F.relu(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(self.BN3(F.relu(self.conv3(x))))
        x = self.pool(self.BN4(F.relu(self.conv4(x))))
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def train_val(model, device, train_loader, val_loader, optimizer, criterion, epoch, writer):
    model.train()
    total_loss = 0.0
    val_loss = 0.0
    val_acc = 0
    for batch_id, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    train_loss = total_loss / len(train_loader.dataset)
    writer.flush()

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, pred = torch.max(outputs, dim=1)
            correct = pred.eq(labels.view_as(pred))
            accuracy = torch.mean(correct.type(torch.FloatTensor))
            val_acc += accuracy.item() * images.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc =100* val_acc / len(val_loader.dataset)
        writer.add_scalar('Val loss', val_loss, epoch)
        writer.add_scalar('Val Acc', val_acc, epoch)
        writer.flush()
    return train_loss, val_loss, val_acc

def test(model, device, test_loader, criterion, epoch, writer):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for batch_id, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
            misclassified_images(predicted, writer, labels, images, outputs, epoch)
        avg_loss = 100 * total_loss / len(test_loader.dataset)
        accuracy = 100 * correct / len(test_loader.dataset)
        writer.add_scalar("Test Loss", avg_loss, epoch)
        writer.add_scalar("Test Acc", accuracy, epoch)
        writer.flush()
        return total_loss, accuracy, avg_loss

def train_epochs(model, device, dataloaders, criterion, optimizer, epochs, writer):
    print("{0:>15} | {1:>15} | {2:>15} | {3:>15} | {4:>15} | {5:>15}  | {6:>15}".format('Epoch', 'Train Loss', 'val_loss', 'val_acc', 'Test Loss', 'Test_acc', 'avg_loss'))
    best_acc = -np.inf
    for epoch in range(epochs):
        train_loss, val_loss, val_acc = train_val(model, device, dataloaders['train'], dataloaders['val'], optimizer, criterion, epoch, writer)
        test_loss, test_acc, avg_loss = test(model, device, dataloaders['test'], criterion, epoch, writer)
        if val_acc > best_acc:
             best_acc = val_acc
             torch.save(model.state_dict(), 'best_val_zhx_SGD.pth')
        print("{0:>15} | {1:>15} | {2:>15} | {3:>15} | {4:>15} | {5:>15}  | {6:>15}".format(epoch, train_loss, val_loss, val_acc, test_loss, test_acc, avg_loss))
        writer.flush()

image_transforms = {
    'train': transforms.Compose([
        transforms.Resize([101, 101]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize([101, 101]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize([101, 101]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
data_dir = 'C:/Users/'
train_dir = data_dir + 'train/'
val_dir = data_dir + 'val/'
test_dir = data_dir + 'test/'
datasets = {
    'train' : datasets.ImageFolder(train_dir, transform=image_transforms['train']),
    'val' : datasets.ImageFolder(val_dir, transform=image_transforms['val']),
    'test' : datasets.ImageFolder(test_dir, transform=image_transforms['test'])
}
BATCH_SIZE = 128
dataloaders = {
    'train' : DataLoader(datasets['train'], batch_size=BATCH_SIZE, shuffle=True),
    'val' : DataLoader(datasets['val'], batch_size=BATCH_SIZE, shuffle=True),
    'test' : DataLoader(datasets['test'], batch_size=BATCH_SIZE, shuffle=True)
}
LABEL = dict((v, k) for k, v in datasets['train'].class_to_idx.items())
log_path = 'logdir/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device.type)
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
writer = tb_writer()
epochs = 1000
train_epochs(model, device, dataloaders, criterion, optimizer, epochs, writer)
writer.close()
