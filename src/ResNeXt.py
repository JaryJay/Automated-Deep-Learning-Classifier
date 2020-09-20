import os
import sys
import time
import winsound

import matplotlib.pyplot as plt
import torch
import torchvision
import xlwt
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

#Notes：
#1. make sure the image folder is at correct path
#2. model is ok or need to freeze the CNNs
#3. training result

def main():
    #input a name for a project
    task_name = r''
    print('performing task:',task_name)
    base_path = base_path = os.path.dirname(os.path.abspath(__file__))
    #relative path
    path_dict = {
        'image_path': r"training_data",  #training set path
        'save_path': r"saved_files",  #model and parameter stored path
        'tensorboard_log_path': r"tensorboard"  #tensorboard for record
    }
    #create a folder
    for key, value in path_dict.items():
        path_dict[key] = os.path.join(base_path, value, task_name)
        if not os.path.exists(path_dict[key]):
            os.mkdir(path_dict[key])

    #run is on CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on:', device)
    torch.cuda.empty_cache()  #clean up GPU memory

    #use the Cal_Mean_Var in Data Preprocessing Folder to calculate the mean and variance
    image_stat_dict = {
        'both': ([0.568212, 0.619295, 0.661991], [0.258661, 0.208433, 0.192903]),
        'positive' : ([0.56447, 0.611874, 0.655232], [0.26677, 0.221352, 0.208195]),
        'negtive' : ([0.572144, 0.627094, 0.669094], [0.249796, 0.193626, 0.175121])
    }
    image_mean, image_std = image_stat_dict['negtive']

    #Data Preprocessing
    image_transforms = {
        'train':
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std)
        ]),
        'val':
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std)
        ])
    }

    data_sets = {
        x:
        torchvision.datasets.ImageFolder(os.path.join(path_dict['image_path'],
                                                      x),
                                         transform=image_transforms[x])
        for x in ['train', 'val']
    }

    class_names = data_sets['train'].classes  #get the categories class name
    total_classes = len(class_names)  #the total nums of the categories
    batchSize = 4  #mini batch size

    data_loaders = {
        x: torch.utils.data.DataLoader(
            data_sets[x],
            batch_size=batchSize,
            pin_memory=True,  
            shuffle=True, 
            num_workers=4)  
        for x in ('train', 'val')
    }

    #Use pre-trained ResNeXt (applied transfer learning)
    model = torchvision.models.resnext101_32x8d(pretrained=True)
    for param in model.parameters():  #use default parameters
        param.requires_grad = False
    #modify last layer
    model.fc = nn.Linear(model.fc.in_features, total_classes, bias=True)
    model.to(device)

    #create the loss function
    criterion = nn.CrossEntropyLoss().to(device)
    #choose the optimizer as Adam
    optimizer = optim.Adam(model.fc.parameters(), lr=5e-5, betas=(0.9, 0.999))

    def trainModel(model, train_loader):
        model.train() 
        train_loss = 0.0  #record the total loss
        for images, labels in train_loader:
            #train it on GPU
            images, labels = images.to(device), labels.to(device)
            # smoothed_labels = torch.full(size=(labels.size(0), total_classes),
            #                             fill_value=0.1 /
            #                             (total_classes - 1)).to(device)
            # smoothed_labels = smoothed_labels.scatter_(dim=1,
            #                         index=torch.unsqueeze(labels, dim=1),
            #                         value=0.9)
            # pytorch with one-hot encoding for training labels
            optimizer.zero_grad()  #clean up optimizer
            output = model(images)  #get the predict result
            batch_loss = criterion(output, labels)  #calculate the loss
            batch_loss.backward()  #backprop
            optimizer.step()  #update parameter
            train_loss += batch_loss.item()  #add the loss to tot loss
        #calculate the avg loss divided by mini-batch nums
        average_train_loss = train_loss / len(train_loader)
        return average_train_loss

    def testModel(model, test_loader):
        model.eval()  
        test_loss, correct_num = 0.0, 0.0  #calculate the loss and total correct nums
        with torch.no_grad():  
            for images, labels in test_loader:  
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                batch_loss = criterion(output, labels)
                test_loss += batch_loss.item()
                prediction = torch.argmax(output, 1)  #find the maximun output nueral
                correct_num += (
                    prediction == labels.data).sum()  #each mini batch correct num
        #calculate the total accuracy
        accuracy = correct_num / len(data_sets['val'])
        accuracy = float(accuracy)
        #calculate the avg test loss
        average_test_loss = test_loss / len(test_loader)
        return average_test_loss, accuracy

    def visualize_data(train_losses, train_times, test_losses, test_accuracy,
                       epoch_num):
        plt.plot(range(epoch_num), train_losses, 'b.-', test_losses, 'r.-')
        plt.title('losses vs. epoches')
        plt.ylabel('Train: bule; Test: red')
        plt.xlabel('Epoches')
        plt.savefig(
            os.path.join(path_dict['save_path'], task_name + "_losses.jpg"))
        plt.close()

        plt.plot(range(epoch_num), test_accuracy, 'g.-')
        plt.title('Test accuracy vs. epoches')
        plt.ylabel('accuracy')
        plt.xlabel('Epoches')
        plt.savefig(
            os.path.join(path_dict['save_path'], task_name + "_accuracy.jpg"))
        plt.close()
        
        plt.plot(range(epoch_num), train_times[0:-1], 'g.-')
        plt.title('Train time vs. epoches')
        plt.ylabel('time (s)')
        plt.xlabel('Epoches')
        plt.savefig(
            os.path.join(path_dict['save_path'], task_name + "_time.jpg"))
        plt.close()

        workBook = xlwt.Workbook(encoding='utf-8')
        loss_sheet = workBook.add_sheet('losses')  
        loss_sheet.write(0, 0, label='epoch')
        loss_sheet.write(0, 1, label='train loss')  
        loss_sheet.write(0, 2, label='test loss') 
        accuracy_sheet = workBook.add_sheet('accuracy') 
        accuracy_sheet.write(0, 0, label='epoch')
        accuracy_sheet.write(0, 1, label='test accuracy')
        time_sheet = workBook.add_sheet('time')  
        time_sheet.write(0, 0, label='epoch')
        time_sheet.write(0, 1, label='run time (s)')
        class_sheet = workBook.add_sheet('classes')  
        for i in range(epoch_num):  
            row = i + 1
            loss_sheet.write(row, 0, label=i)
            loss_sheet.write(row, 1, label=train_losses[i])
            loss_sheet.write(row, 2, label=test_losses[i])
            accuracy_sheet.write(row, 0, label=i)
            accuracy_sheet.write(row, 1, label=test_accuracy[i])
            time_sheet.write(row, 0, label=i)
            time_sheet.write(row, 1, label=train_times[i])
        for index, name in enumerate(class_names):
            class_sheet.write(index, 0, label=name)
       
        workBook.save(
            os.path.join(path_dict['save_path'],
                         task_name + '_running_data.xls'))

   
    log_path = os.path.join(path_dict['save_path'], 'logs')
    if os.path.exists(log_path):  
        latest = max(os.listdir(log_path))  
        check_point = torch.load(os.path.join(log_path, latest),
                                 map_location=device)
        model.load_state_dict(check_point['model_state_dict'])
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
        start_epoch = check_point['epoch'] + 1
        
        train_losses = check_point['train_losses']
        train_times = check_point['train_times']
        test_losses = check_point['test_losses']
        test_accuracy = check_point['test_accuracy']
        lowest_loss = max(test_losses)  
        best_model_weights = check_point['best_model_weights']
        print('succeded in loading the {}th epoch!'.format(start_epoch))
    else:  
        start_epoch = 0
        lowest_loss = 3.0
        train_losses, train_times, test_losses, test_accuracy = [], [], [], []
        print('no check point found, start from the first epoch!')

    epoch_num = 64  #epoch num
    save_every = 8 # save model
    since = time.time()  
    
    writer = SummaryWriter(log_dir=path_dict['tensorboard_log_path'])
    for epoch in range(start_epoch, epoch_num):  
        try:
            print('-' * 10)
            print('Epoch {}/{} :'.format(epoch, epoch_num - 1))
            print(time.strftime("%Y/%m/%d %H: %M: %S", time.localtime()))
            start = time.time()  
            average_train_loss = trainModel(model, data_loaders['train'])  
            train_losses.append(average_train_loss)  
            average_test_loss, accuracy = testModel(model,
                                                    data_loaders['val'])  
            test_losses.append(average_test_loss) 
            test_accuracy.append(accuracy)  
            print(
                'train loss: {:.4f}; test loss: {:.4f}; accuracy : {:.4f}'.format(
                    average_train_loss, average_test_loss, accuracy))
            time_elapsed = time.time() - start  
            train_times.append(time_elapsed)  
            print('epoch training completed in: {:.3f} seconds'.format(
                time_elapsed))

        
            writer.add_scalars('losses', {
                'train loss': average_train_loss,
                'test loss': average_test_loss
            }, epoch)
            writer.add_scalar('test accuracy', accuracy, epoch)
            writer.add_scalar('training time', time_elapsed, epoch)

            if average_test_loss < lowest_loss and accuracy > 0.913: 
                print("new lost record: ", average_test_loss)
                #update the greatest loss
                lowest_loss = average_test_loss
                #remember the best weights
                best_model_weights = model.state_dict()

            if epoch % save_every == save_every - 1:
                check_point_name = time.strftime(
                    "%Y%m%d%H%M%S", time.localtime()) + '_' + str(epoch) + '.tar'
                if not os.path.exists(log_path):  
                    os.mkdir(log_path)
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'train_losses': train_losses,
                        'train_times': train_times,
                        'test_losses': test_losses,
                        'test_accuracy': test_accuracy,
                        'best_model_weights': best_model_weights
                    }, os.path.join(log_path, check_point_name))
                print(check_point_name, 'saved!')
                CoffinDance(0)
        except KeyboardInterrupt:  
            print('KeyboardInterrupted!')
            check_point_name = time.strftime(
                "%Y%m%d%H%M%S", time.localtime()) + '_' + str(epoch) + '.tar'
            if not os.path.exists(log_path):  
                os.mkdir(log_path)
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_losses': train_losses,
                    'train_times': train_times,
                    'test_losses': test_losses,
                    'test_accuracy': test_accuracy,
                    'best_model_weights': best_model_weights
                }, os.path.join(log_path, check_point_name))
            choice = input("tap 'yes' to continue, 'stop' to exit the programme!")
            if choice == 'stop':
                sys.exit(0)

    
    torch.save(
        {
            'model': model,
            'parameters': best_model_weights,
            'classes': class_names
        },
        os.path.join(path_dict['save_path'],
                    task_name + '_model_param_classes.pth'))
    
    time_elapsed = time.time() - since
    train_times.append(time_elapsed)
    print('Training completed in: ', time_elapsed, 'seconds')
    
    writer.close()
    
    visualize_data(train_losses, train_times, test_losses, test_accuracy,
                epoch_num)


def CoffinDance(repeat=3):
    q = 1.06
    piano2 = {'C': 523,'D':587,'E':659,'F':698,'G':784,'A':880,'B':988}
    piano = {'C':261,'D':293,'E':329,'F':349,'G':392,'A':440,'B':494}
    note = [0]
    do = 'F'
    note.append(piano[do])
    c = piano[do]
    note.append(int(c*q**2))
    note.append(int(c*q**4))
    note.append(int(c*q**5))
    note.append(int(c*q**7))
    note.append(int(c*q**9))
    note.append(int(c*q**11))
    song = ['4N', '4N', '4N', '4N', '6N', '6N', '6N', '6N', 
            '5N', '5N', '5N', '5N', '1H', '1H', '1H', '1H', 
            '2H', '2H', '2H', '2H', '2H', '2H', '2H', '2H', 
            '5N', '4N', '3N', '1N', '2N', '0N', '2N', '6N', 
            '5N', '0N', '4N', '0N', '3N', '0N', '3N', '3N', 
            '5N', '0N', '4N', '3N', '2N', '0N', '2N', '4H', 
            '3H', '4H', '3H', '4H', '2N', '0N', '2N', '4H', 
            '3H', '4H', '3H', '4H', '2N', '0N', '2N', '6N', 
            '5N', '0N', '4N', '0N', '3N', '0N', '3N', '3N', 
            '5N', '0N', '4N', '3N', '2N', '0N', '2N', '4H', 
            '3H', '4H', '3H', '4H', '2N', '0N', '2N', '4H', 
            '3H', '4H', '3H', '4H', '4N', '4N', '4N', '4N', 
            '6N', '6N', '6N', '6N', '5N', '5N', '5N', '5N', 
            '1H', '1H', '1H', '1H', '2H', '2H', '2H', '2H', 
            '2H', '2H', '2H', '2H', '5N', '4N', '3N', '1N', 
            '2N', '0N', '2N', '6N', '5N', '0N', '4N', '0N', 
            '3N', '0N', '3N', '3N', '5N', '0N', '4N', '3N', 
            '2N', '0N', '2N', '4H', '3H', '4H', '3H', '4H', 
            '2N', '0N', '2N', '4H', '3H', '4H', '3H', '4H', 
            '2N', '0N', '2N', '6N', '5N', '0N', '4N', '0N', 
            '3N', '0N', '3N', '3N', '5N', '0N', '4N', '3N', 
            '2N', '0N', '2N', '4H', '3H', '4H', '3H', '4H', 
            '2N', '0N', '2N', '4H', '3H', '4H', '3H', '4H', 
            '2N', '0N', '2N', '6N', '5N', '0N', '4N', '0N', 
            '3N', '0N', '3N', '3N', '5N', '0N', '4N', '3N', 
            '4N', '4N', '4N', '4N', '6N', '6N', '6N', '6N', 
            '5N', '5N', '5N', '5N', '1H', '1H', '1H', '1H', 
            '2H', '2H', '2H', '2H', '2H', '2H', '2H', '2H', 
            '0N', '0N', '0N', '0N', '0N', '0N', '0N', '0N',
    ]
    if repeat < 1:
        return
    for i in range(repeat):
        for each in song:
            a = int(each[0])
            b = each[1]
            if a == 0:
                time.sleep(0.238)
            else:
                if b == 'N':
                    winsound.Beep(note[a],238)
                else:
                    winsound.Beep(2*note[a],238)

if __name__ == '__main__':
    main()
    CoffinDance(0)