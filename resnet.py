
# TODO: Saving transformed imgs in seperate folder and with postfix
# TODO: Option for calculating test accuracy from last saved checkpoint
# TODO: Multiprocessing Support
# TODO: Multi GPU Support
# TODO: Optimize testing and validation accurayc calculation shift to cpu and numpy
# TODO: Confusion matrix


import os
from random import shuffle
import torch
import tools
import models
import random
import torchvision
import numpy as np
import pandas as pd
import seaborn as sb
from sched import scheduler
from torchsummary import summary
from torch.autograd import Variable
from matplotlib import pyplot as plt
from argparse import ArgumentParser as Args
from sklearn.metrics import confusion_matrix
from transformers import DataTransformer, Transforms


def main():
    
    # Argument Parser
    parser = Args()
    parser.add_argument("-d",   "--data_path",          type=str, default="CIFAR10/",
                        metavar="<path>",
                        help="Dataset storage path | default: CIFAR10/")
    parser.add_argument("-trb", "--train_batch_size",   type=int, default=128,
                        metavar="<int>",
                        help="Batch size for training | default: 128")
    parser.add_argument("-vb",  "--val_batch_size",     type=int, default=256,
                        metavar="<int>",
                        help="Batch size for validation | default: 256")
    parser.add_argument("-tsb", "--test_batch_size",    type=int, default=256,
                        metavar="<int>",
                        help="Batch size for testing | default: 256")
    parser.add_argument("-e",   "--epochs",             type=int, default=1,
                        metavar="<int>",
                        help="Number of epochs for training | default: 1")
    parser.add_argument("-tsf", "--test_transform",     type=str, default="TestTransform_1",
                        metavar="<str>",
                        help="Transfom to be applied on Testing Data | default: TestTransform_1")
    parser.add_argument("-trf", "--train_transform",    type=str, default="TrainTransform_1",
                        metavar="<str>",
                        help="Transfom to be applied on Training Data | default: TrainTransform_1")
    parser.add_argument("-r",   "--resume",             type=str,
                        metavar="<str>", help="Load model and Resume Trainig from given filename")
    parser.add_argument("-m",   "--model",              type=str, default="ResNet10_1",
                        metavar="<str>",
                        help="ResNet Model | default: ResNet10_1")                
    parser.add_argument("-o",   "--optimizer",          type=str, default='sgd',
                        choices=['sgd','sgdn','adagrad', 'adadelta','adam'],
                        metavar="<str>", help="Optimizer for the network | \
                        choices: sgd , sgdn, adagrad , adadelta , adam | \
                        default: sgd")
    parser.add_argument("-ol",  "--learning_rate",      type=float, default=0.1,
                        metavar="<float>",
                        help="Learning Rate of optimizer | default: 0.1")
    parser.add_argument("-om",  "--momentum",           type=float, default=0.9,
                        metavar="<float>",
                        help="Momentum of optimizer | default: 0.9")
    parser.add_argument("-ow",  "--weight_decay",       type=float, default=5e-4,
                        metavar="<float>",
                        help="Weight Decay of optimizer | default: 5e-4")
    parser.add_argument("-mx",  "--mixup",              action="store_true", 
                        help="Enable Mixup Augmentaion | default: false")
    parser.add_argument("-sc",  "--schedule",           action="store_true", 
                        help="Enable Optimizer LR Decay sceduling | default: false")
    parser.add_argument("-an",  "--anneal",             action="store_true", 
                        help="Enable Optimizer Annealing | default: false")
    parser.add_argument("-v",   "--validate",           action="store_true", 
                        help="Enable Validation alongside Testing| default: false")
    parser.add_argument("-s",   "--save",               action="store_true", 
                        help="Save best performaing model | default: false")
    parser.add_argument("-ss",   "--save_state",        action="store_true", 
                        help="Save best model state dict | default: false")
    parser.add_argument("-p",   "--print_summary",      action="store_true", 
                        help="Print Network Summary | default: false")
    parser.add_argument("-n",   "--no_training",        action="store_true", 
                        help="No training will take place | default: false")
    parser.add_argument("-sp",  "--save_plot",          action="store_true", 
                        help="Error Plot will be saved | default: false")
    parser.add_argument("-pt",  "--plot_transforms",    action="store_true", 
                        help="Save plots for After and Before applying \
                            Transforms | default: false")
    parser.add_argument("-pcm",  "--plot_conf_mat",    action="store_true", 
                        help="Save confusion matrix plot | default: false")
    parser.add_argument("-ofp", "--output_file_prefix", type=str, default="ResNet",
                        metavar="<str>",
                        help="Output file name prefix for saving model state and Error plot \
                            file | default: ResNet")
    args = parser.parse_args()    


    # Print Run Config
    print("\n\nRun Config: \n------------------------------")
    for key in args.__dict__:
        print("{:<18} : {}".format(key,args.__dict__[key]))

    # Set random seed
    seed = 43
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Device selection
    if torch.cuda.is_available:
        use_cuda = True
        device = "cuda" 
        print("\nRunning on ",device.upper(),"\n")
        print("Number of GPUs:",torch.cuda.device_count())
        print("Cuda Capability:",torch.cuda.get_device_capability(),'\n')
    else:
        device="cpu"
    


    # Data Download
    download_dir = args.data_path
    trainData = torchvision.datasets.CIFAR10(download_dir, 
                        train=True ,download=True,
                        transform=None)
    testData  = torchvision.datasets.CIFAR10(download_dir, 
                        train=False,download=True,
                        transform=None)

    # DEBUG
    # print('Train Data size:',len(trainData))
    # print('Test Data size:',len(testData))


    # Create Validation Dataset
    if args.validate:
        len_ = len(trainData)
        trainData, valData = torch.utils.data.dataset.random_split(
            trainData, [round(len_*0.9), round(len_*0.1)])


    # Save Images Plots
    if args.plot_transforms:
        tools.plotImg(trainData,5,"Train")
        tools.plotImg(testData,5,"Test")


    # Apply Data Transforms
    try:
        exec('args.train_transform = Transforms.{}()'.format(args.train_transform))
        train_transform = args.train_transform
    except Exception as e:
        print('\n' + str(e))
        print("\nError importing Train Transform\n\nExiting...\n")
        exit(0)
    
    try:
        exec('args.test_transform = Transforms.{}()'.format(args.test_transform))
        test_transform = args.test_transform
        val_transform = args.test_transform
    except Exception as e:
        print('\n' + str(e))
        print("\nError importing Test Transform\n\nExiting...\n")
        exit(0)
    
    trainData = DataTransformer(trainData,train_transform)
    testData = DataTransformer(testData,test_transform)
    
    if args.validate:
        valData = DataTransformer(valData,val_transform)
    else:
        valData = testData


    # Save Image plots after Data Augmentation
    if args.plot_transforms:
        tools.plotImg(trainData,5,"Train_transformed",True)
        tools.plotImg(testData,5,"Test_transformed",True)


    # Data Loaders
    workers = 2      # 2 works best for CIFAR10
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    trainDataLoader = torch.utils.data.DataLoader(trainData, num_workers=workers,
                        batch_size=train_batch_size, shuffle=True )
    valDataLoader  = torch.utils.data.DataLoader(valData, num_workers=workers, 
                        batch_size=val_batch_size, shuffle=False)
    if args.validate:
        test_batch_size = args.val_batch_size
        testDataLoader  = torch.utils.data.DataLoader(testData, num_workers=workers, 
                        batch_size=test_batch_size, shuffle=False)
        

    # Network Definiton
    if args.resume:
        # Load checkpoint.
        print('\n\nResuming from checkpoint..\n')
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoints/' + args.resume + '_state.pt')
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        train_loss_history = checkpoint['train_loss_history']
        val_loss_history = checkpoint['val_loss_history']
        optimizer = checkpoint['optimizer']
        rng_state = checkpoint['rng_state']
        torch.set_rng_state(rng_state)
    else:
        best_acc = 0
        start_epoch = 0
        train_loss_history = []
        val_loss_history  = []
        try:
            exec('args.model = models.{}().to("{}")'.format(args.model,device))
            net = args.model
        except Exception as e:
            print('\n' + str(e))
            print("\nError importing Model.\n\nExiting...\n")
            exit(0)

    # Set Data Parallelism
    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    # Loss Definition
    Loss = torch.nn.CrossEntropyLoss()


    # Model Optimizer
    lr = args.learning_rate
    m = args.momentum
    wd = args.weight_decay
    if not args.resume:
        if   args.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=m, weight_decay=wd)
        elif args.optimizer.lower() == "sgdn":
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=m, weight_decay=wd, nesterov=True)
        elif args.optimizer.lower() == "adagrad":
            optimizer = torch.optim.Adagrad(net.parameters(), lr=lr, weight_decay=wd)
        elif args.optimizer.lower() == "adadelta":
            optimizer = torch.optim.Adadelta(net.parameters(), lr=lr, weight_decay=wd)
        elif args.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)


    # Enable Annealing Scheduler
    if args.anneal:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # Model Summary
    if args.print_summary:        
        print("\n\nPrinting Model Summary...\n")
        # DEBUG
        # modules = [module for module in net.modules()]
        # print(modules[0])
        summary(net, input_size=(3,32,32))
        print()


    # Exit if no training 
    if args.no_training:
        print("\nExiting...\n")
        exit(1)

    # Training Function
    def train():
        train_loss = 0.0
        train_acc = 0.0
        net.train()
        for batch, data in enumerate(trainDataLoader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = Loss(outputs,labels)
            loss.backward()
            optimizer.step()

            # DEBUG
            # os.system('nvidia-smi')

            train_loss += loss.item()
            predicted = torch.argmax(outputs,dim=1)
            train_acc += predicted.eq(labels).cpu().sum().item()

        train_acc = 100.0 * train_acc / len(trainData)
            
        return (train_loss, train_acc)

    # Validation Function
    def validate():
        val_loss = 0.0
        val_acc = 0.0
        net.eval()
        with torch.no_grad():
            for batch, data in enumerate(valDataLoader):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                outputs = net(images)
                loss = Loss(outputs, labels)

                val_loss += loss.item()
                predicted = torch.argmax(outputs,dim=1)
                val_acc += predicted.eq(labels).cpu().sum().item()

        val_acc = 100.0 * val_acc / len(valData)

        return (val_loss, val_acc)

    # Mixeup Helper Functions
    def mixup_data(x, y, alpha=1.0, use_cuda=True):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    def mixup_Loss(Loss, pred, y_a, y_b, lam):
        return lam * Loss(pred, y_a) + (1 - lam) * Loss(pred, y_b)

    # Mixup train
    def mixupTrain():
        train_loss = 0.0
        train_acc = 0.0
        net.train()
        alpha = 1
        for batch, data in enumerate(trainDataLoader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha)
            images, targets_a, targets_b = map(Variable, (images, targets_a, targets_b))

            optimizer.zero_grad()
            outputs = net(images)
            loss = mixup_Loss(Loss, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

            # DEBUG
            # os.system('nvidia-smi')

            train_loss += loss.item()
            predicted = torch.argmax(outputs,dim=1)
            train_acc += predicted.eq(labels).cpu().sum().item()

        train_acc = 100.0 * train_acc / len(trainData)
            
        return (train_loss, train_acc)


    # Model Training
    epochs = args.epochs
    print("\nTraining...\n\n")
    for epoch in range(epochs):

        train_acc = 0.0
        val_acc = 0.0

        # Train Loop
        if args.mixup:
            train_loss, train_acc = mixupTrain()
        else:
            train_loss, train_acc = train()
        
        # Validation Loop
        val_loss, val_acc = validate()

        # Annealing Scheduler Step
        if args.anneal:
            scheduler.step()

        # Learning Rate Decay
        if args.schedule:
            if epoch+start_epoch >= 100:
                lr /= 10
            if epoch+start_epoch >= 150:
                lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        train_loss = train_loss / len(trainDataLoader)
        val_loss = val_loss / len(valDataLoader)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print('Epoch: %s ;  Train Loss: %.8f ;  Train Accuracy: %.3f ;  Val Loss: %.8f ;  Val Accuracy: %.3f ;'\
            %(start_epoch+epoch+1, train_loss, train_acc, val_loss, val_acc))
        
        # Save checkpoint
        if val_acc > best_acc:
            if args.save:
                state = {
                    'net': net,
                    'acc': val_acc,
                    'epoch': start_epoch + epoch,
                    'train_loss_history' : train_loss_history,
                    'val_loss_history' : val_loss_history,
                    'optimizer' : optimizer,
                    'rng_state': torch.get_rng_state()
                }
                if not os.path.isdir('checkpoints'):
                    os.mkdir('checkpoints')
                torch.save(state, './checkpoints/'+args.output_file_prefix+'_'+str(epochs+start_epoch)+'_state.pt')
                print('\nNetwork State saved ('+args.output_file_prefix+'_'+str(epochs+start_epoch)+'_state.pt)\n')
            best_acc = val_acc


    # Load Best network state to calculate Test Metrics
    if args.resume or args.save:
        last_best = './checkpoints/'+args.output_file_prefix+'_'+str(epochs+start_epoch)+'_state.pt'
        resumed_from = './checkpoints/' + args.resume + '_state.pt'
        if os.path.exists(last_best):
            checkpoint = torch.load(last_best)
        elif os.path.exists(resumed_from):
            checkpoint = torch.load(resumed_from)
        if checkpoint:
            print('\nBest Network state Loaded')
            net = checkpoint['net']
            optimizer = checkpoint['optimizer']
            rng_state = checkpoint['rng_state']
            torch.set_rng_state(rng_state)


    # Save Best Model State Dict
    if args.save_state:
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(net.state_dict(), './checkpoints/project1_model.pt')
        print('\nNetwork State dict saved (./checkpoints/project1_model.pt)\n')


    # Test Loop
    if args.validate or args.plot_conf_mat:
        test_loss = 0.0
        test_acc = 0.0
        predictions = []
        targets = []
        net.eval()
        with torch.no_grad():
            for batch, data in enumerate(testDataLoader):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                outputs = net(images)
                loss = Loss(outputs, labels)

                test_loss += loss.item()
                predicted = torch.argmax(outputs,dim=1)
                test_acc += predicted.eq(labels).sum().item()
                
                if args.plot_conf_mat:
                    predictions.extend(predicted.cpu())
                    targets.extend(labels.cpu())

        test_acc = 100.0 * test_acc / len(testData)
        test_loss = test_loss / len(testDataLoader)

        print('\nEpochs: %s ;\tTest Loss: %.8f ; \tTest Accuracy: %.3f ;'\
            %(start_epoch+epochs, test_loss, test_acc))

        # Save Confusion Matrix
        if args.plot_conf_mat:
            classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            cf_matrix = confusion_matrix(targets,predictions)
            df_cm = pd.DataFrame(cf_matrix,index = [i for i in classes],
                                 columns = [i for i in classes])
            plt.figure(figsize=(6,6));
            sb.heatmap(df_cm, annot=True,fmt="d",linewidths=.5,cmap=["#FDBFBF",'#74BDFF'], cbar=False);
            plt.savefig('./plots/'+args.output_file_prefix + '_confusion_matrix.png');
            print('\n\nConfusion Matrix saved ('+args.output_file_prefix+'_confusion_matrix.png)')

    # Print Best Accuracy of the run
    print("\nBest Accuracy: %.3f\n"%(best_acc))


    # Save Error Plot
    if args.save_plot:
        print('\nSaving Error Plot...\n')
        plt.figure(figsize=[12, 5]);
        plt.plot(range(1,start_epoch+epochs+1),train_loss_history,'-',linewidth=3,label='Train error');
        plt.plot(range(1,start_epoch+epochs+1),val_loss_history ,'-',linewidth=3,label='Val error');
        plt.xlabel('epoch',fontsize=13);
        plt.ylabel('loss' ,fontsize=13);
        plt.grid(True);
        plt.legend();
        # plt.xticks(range(1,start_epoch+epochs+1));
        plt.title('Error Plot',  fontsize=15);
        if not os.path.isdir('plots'):
            os.mkdir('plots')
        plt.savefig('./plots/'+args.output_file_prefix + '_error_plot.png');
        print('Error Plot saved ('+args.output_file_prefix+'_error_plot.png)\n')

 
if __name__ == "__main__":
    main()