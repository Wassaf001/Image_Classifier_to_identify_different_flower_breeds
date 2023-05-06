import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

def main():
    i_arg = get_input_args()

    arch = i_arg.arch
    data_dir = i_arg.data_directory
    epochs = i_arg.epochs
    gpu = i_arg.gpu
    hidden_units = i_arg.hidden_units
    learn_rate = i_arg.learning_rate
    save_dir = i_arg.save_dir

    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    batch_size = 64
    print_every = 20
    input_size = int(hidden_units)
    output_size = 102

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    model_to_user = getattr(models, arch)
    model = model_to_user(pretrained = True)

    for param in model.parameters():
        param.requires_grad = False

    fc1_in = input_size
    fc1_out = input_size // 2
    fc2_in = fc1_out
    fc2_out = fc2_in // 2
    fc3_in = fc2_out
    fc3_out = output_size

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(fc1_in, fc1_out)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p = 0.5)),
        ('fc2', nn.Linear(fc2_in, fc2_out)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(fc3_in, fc3_out)),
        ('output', nn.LogSoftmax(dim = 1))
    ]))
    
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    do_deep_learning(model, trainloader, testloader, epochs, print_every, criterion, optimizer, device)

    checkpoint = {
        'epochs': epochs,
        'input_size': input_size,
        'output_size': output_size,
        'learn_rate': learn_rate,
        'batch_size': batch_size,
        'data_transforms': data_transforms,
        'model': model_to_user(pretrained = True),
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'class_to_idx': train_data.class_to_idx,
        'classifier': model.classifier,
    }
    torch.save(checkpoint, './' + save_dir + '/checkpoint.pth')

def get_input_args():
   
    parser = argparse.ArgumentParser(description = 'Training NN options')

    parser.add_argument(
        'data_directory',
        action = 'store',
        help = 'path to directory of data'
    )
    parser.add_argument(
        '--save_dir',
        type = str,
        dest = 'save_dir',
        default = '',
        help = 'path to directory to save the checkpoint'
    )
    parser.add_argument(
        '--arch',
        type = str,
        dest = 'arch',
        default = 'densenet121',
        help = 'chosen model (any model on \'torchvision.models\', please insert as value the exact name of the model => densenet121, vgg13...)'
    )
    parser.add_argument(
        '--learning_rate',
        type = float,
        dest = 'learning_rate',
        default = 0.001,
        help = 'chosen learning rate'
    )
    parser.add_argument(
        '--hidden_units',
        type = int,
        dest = 'hidden_units',
        default = 1024,
        help = 'chosen hidden units, this value is dependant of the choosen model'
    )
    parser.add_argument(
        '--epochs',
        type = int,
        dest = 'epochs',
        default = 10,
        help = 'chosen epochs'
    )
    parser.add_argument(
        '--gpu',
        dest = 'gpu',
        action = 'store_true',
        help = 'use the GPU for training'
    )
    return parser.parse_args()

def do_deep_learning(model, trainloader, testloader, epochs, print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    steps = 0
    model = model.to(device)
    model.train()
    print("Training model...")
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                accuracy = 0
                validation_loss  = 0
                for ii, (inputs, labels) in enumerate(testloader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model.forward(inputs)
                    validation_loss  += criterion(output, labels)
                    probabilities = torch.exp(output).data
                    equality = (labels.data == probabilities.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "| Training Loss: {:.4f}".format(running_loss / print_every),
                      "| Validation Loss: {:.3f}.. ".format(validation_loss  / len(testloader)),
                      "| Validation Accuracy: {:.3f}%".format(accuracy / len(testloader) * 100))
                running_loss = 0
                model.train()
    print("Done!")

if __name__ == '__main__':
    main()