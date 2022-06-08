import torch
import random
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import DHR_Net as models
import numpy as np
import pickle
import os
from tqdm import tqdm
import customized_dataloader
from customized_dataloader import msd_net_dataset



###################################################################
                            # options #
###################################################################
train_phase = False

seed = 4
best_epoch = 199

batch_size = 64
img_size = 32

lr = 0.05
epochs = 200
momentum = 0.9
nb_classes = 293
weight_decay = 0.0005

result_dir ="/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24/models/crosr/"
save_model_path = result_dir + "/seed_" + str(seed)
save_result_path = result_dir + "/seed_" + str(seed) + "/train_valid_results.txt"
test_model_path = save_model_path + "/model_epoch_" + str(best_epoch) + ".dat"
feature_save_path = save_model_path + "/features/"

#####################################################################
            # Paths for saving model and data source #
#####################################################################
json_data_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                 "dataset_v1_3_partition/npy_json_files_shuffled/"

train_known_known_path = os.path.join(json_data_base, "train_known_known.json")
valid_known_known_path = os.path.join(json_data_base, "valid_known_known.json")

test_known_known_path_p0 = os.path.join(json_data_base, "test_known_known_part_0.json")
test_known_known_path_p1 = os.path.join(json_data_base, "test_known_known_part_1.json")
test_known_known_path_p2 = os.path.join(json_data_base, "test_known_known_part_2.json")
test_known_known_path_p3 = os.path.join(json_data_base, "test_known_known_part_3.json")

test_unknown_unknown_path = os.path.join(json_data_base, "test_unknown_unknown.json")

#######################################################################
# Create dataset and data loader
#######################################################################
# Data transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([transforms.RandomResizedCrop(img_size),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      normalize])

valid_transform = train_transform

test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(img_size),
                                     transforms.ToTensor(),
                                     normalize])

#######################################################################
# Create dataset and data loader
#######################################################################
# Training
train_known_known_dataset = msd_net_dataset(json_path=train_known_known_path,
                                            transform=train_transform)
train_known_known_index = torch.randperm(len(train_known_known_dataset))
train_loader = torch.utils.data.DataLoader(train_known_known_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           drop_last=True,
                                           collate_fn=customized_dataloader.collate,
                                           sampler=torch.utils.data.RandomSampler(
                                               train_known_known_index))

print(len(train_loader))

# Validation
valid_known_known_dataset = msd_net_dataset(json_path=valid_known_known_path,
                                            transform=valid_transform)
valid_known_known_index = torch.randperm(len(valid_known_known_dataset))
valid_loader = torch.utils.data.DataLoader(valid_known_known_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               collate_fn=customized_dataloader.collate,
                                               sampler=torch.utils.data.RandomSampler(
                                                   valid_known_known_index))

# Testing
test_known_known_dataset_p0 = msd_net_dataset(json_path=test_known_known_path_p0,
                                               transform=test_transform)
test_known_known_index_p0 = torch.randperm(len(test_known_known_dataset_p0))

test_known_known_dataset_p1 = msd_net_dataset(json_path=test_known_known_path_p1,
                                              transform=test_transform)
test_known_known_index_p1 = torch.randperm(len(test_known_known_dataset_p1))

test_known_known_dataset_p2 = msd_net_dataset(json_path=test_known_known_path_p2,
                                              transform=test_transform)
test_known_known_index_p2 = torch.randperm(len(test_known_known_dataset_p2))

test_known_known_dataset_p3 = msd_net_dataset(json_path=test_known_known_path_p3,
                                              transform=test_transform)
test_known_known_index_p3 = torch.randperm(len(test_known_known_dataset_p3))


test_unknown_unknown_dataset = msd_net_dataset(json_path=test_unknown_unknown_path,
                                               transform=test_transform)
test_unknown_unknown_index = torch.randperm(len(test_unknown_unknown_dataset))

# When doing test, set the batch size to 1 to test the time one by one accurately
test_known_known_loader_p0 = torch.utils.data.DataLoader(test_known_known_dataset_p0,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      sampler=torch.utils.data.RandomSampler(
                                                          test_known_known_index_p0),
                                                      collate_fn=customized_dataloader.collate,
                                                      drop_last=True)

test_known_known_loader_p1 = torch.utils.data.DataLoader(test_known_known_dataset_p1,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         sampler=torch.utils.data.RandomSampler(
                                                             test_known_known_index_p1),
                                                         collate_fn=customized_dataloader.collate,
                                                         drop_last=True)

test_known_known_loader_p2 = torch.utils.data.DataLoader(test_known_known_dataset_p2,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         sampler=torch.utils.data.RandomSampler(
                                                             test_known_known_index_p2),
                                                         collate_fn=customized_dataloader.collate,
                                                         drop_last=True)

test_known_known_loader_p3 = torch.utils.data.DataLoader(test_known_known_dataset_p3,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         sampler=torch.utils.data.RandomSampler(
                                                             test_known_known_index_p3),
                                                         collate_fn=customized_dataloader.collate,
                                                         drop_last=True)

test_unknown_unknown_loader = torch.utils.data.DataLoader(test_unknown_unknown_dataset,
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          sampler=torch.utils.data.RandomSampler(
                                                              test_unknown_unknown_index),
                                                          collate_fn=customized_dataloader.collate,
                                                          drop_last=True)



def save_features(net,
                  dataloader,
                  data_name,
                  npy_save_dir):
    """

    :param net:
    :param dataloader:
    :return:
    """
    net.eval()

    full_label_list = []
    full_logits_list = []

    with torch.no_grad():
        for i in tqdm(range(len(dataloader))):
            batch = next(dataloader.__iter__())

            # get the inputs; data is a list of [inputs, labels]
            inputs = batch["imgs"]
            inputs = inputs.cuda(non_blocking=True)
            inputs = inputs[:, :, :32, :]

            labels = batch["labels"]
            labels = labels.cuda(non_blocking=True).type(torch.long)

            logits, reconstruct, _ = net(inputs)

            # Save label and logits
            label_list = np.array(labels.cpu().tolist())
            for label in label_list:
                full_label_list.append(label)

            logit_list = np.array(logits.cpu().tolist())
            for logit in logit_list:
                full_logits_list.append(logit)

    label_list_np = np.asarray(full_label_list)
    logit_list_np = np.asarray(full_logits_list)

    np.save(npy_save_dir + data_name + "_logits.npy", logit_list_np)
    np.save(npy_save_dir + data_name + "_labels.npy", label_list_np)




def epoch_train(net,
                trainloader,
                optimizer):
    net.train() 

    correct=0
    total=0
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reconst_loss = 0.0
    iter=0
    cls_criterion = nn.CrossEntropyLoss()
    reconst_criterion = nn.MSELoss()

    for i in tqdm(range(len(train_loader))):
    # for i in tqdm(range(10)):
        batch = next(trainloader.__iter__())

        # get the inputs; data is a list of [inputs, labels]
        inputs = batch["imgs"]
        inputs = inputs.cuda(non_blocking=True)
        inputs = inputs[:, :, :32, :]

        labels = batch["labels"]
        labels = labels.cuda(non_blocking=True).type(torch.long)
    
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        logits, reconstruct,_ = net(inputs)
        cls_loss = cls_criterion(logits, labels)

        reconst_loss = reconst_criterion(reconstruct,inputs)
      
        if(torch.isnan(cls_loss) or torch.isnan(reconst_loss)):
            print("Nan at iteration ",iter)
            cls_loss=0.0
            reconst_loss=0.0
            logits=0.0          
            reconstruct = 0.0  
            continue

        loss = cls_loss + reconst_loss

        loss.backward()
        optimizer.step()  

        total_loss = total_loss + loss.item()
        total_cls_loss = total_cls_loss + cls_loss.item()
        total_reconst_loss = total_reconst_loss + reconst_loss.item()

        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        iter = iter + 1

    return [(100 * (correct / total)), (total_cls_loss/iter), (total_reconst_loss/iter), (total_loss/iter)]




def epoch_val(net,
              testloader):

    net.eval()

    correct = 0
    total = 0
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reconst_loss = 0.0
    iter=0
    cls_criterion = nn.CrossEntropyLoss()
    reconst_criterion = nn.MSELoss()

    with torch.no_grad():
        for i in tqdm(range(len(valid_loader))):
            batch = next(testloader.__iter__())

            # get the inputs; data is a list of [inputs, labels]
            inputs = batch["imgs"]
            inputs = inputs.cuda(non_blocking=True)
            inputs = inputs[:, :, :32, :]

            labels = batch["labels"]
            labels = labels.cuda(non_blocking=True).type(torch.long)

            logits, reconstruct,_ = net(inputs)

            cls_loss = cls_criterion(logits, labels)

            reconst_loss = reconst_criterion(reconstruct, inputs)
        
            loss = cls_loss + reconst_loss

            total_loss = total_loss + loss.item()
            total_cls_loss = total_cls_loss + cls_loss.item()
            total_reconst_loss = total_reconst_loss + reconst_loss.item()

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            iter = iter + 1

    return [(100 * (correct / total)), (total_cls_loss/iter), (total_reconst_loss/iter), (total_loss/iter)]
                 


def main():
    if train_phase:
        print("Training and validating models")
        # Setup random seed
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        net = models.DHRNet(nb_classes)
        net = torch.nn.DataParallel(net.cuda())

        optimizer = optim.SGD(net.parameters(),
                              lr=lr,
                              momentum=momentum,
                              weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=30,
                                              gamma=0.5)

        best_valid_acc = 0.0000

        with open(save_result_path, 'w') as f:
            for epoch in range(epochs):
                train_acc = epoch_train(net,
                                        train_loader,
                                        optimizer)

                valid_acc = epoch_val(net,
                                      valid_loader)
                scheduler.step()

                print("Train accuracy and cls, reconstruct and total loss for epoch "+ str(epoch) +" is "+ str(train_acc))
                print("Valid accuracy and cls, reconstruct and total loss for epoch "+ str(epoch) +" is "+ str(valid_acc))

                f.write('Epoch: [{0}]\t'
                        'Train Acc {train:.4f}\t'
                        'Valid Acc {valid:.4f}\n'.format(epoch,
                                                        train=train_acc[0],
                                                        valid=valid_acc[0]))

                if valid_acc[0] > best_valid_acc:
                    torch.save(net.state_dict(), save_model_path + "/model_epoch_" + str(epoch) + '.dat')
                    torch.save(optimizer.state_dict(), save_model_path + "/optimizer_epoch_" + str(epoch) + '.dat')

    else:
        print("Testing models and saving features")
        net = models.DHRNet(nb_classes)
        net = torch.nn.DataParallel(net.cuda())
        net.load_state_dict(torch.load(test_model_path))
        print("Best model loaded")

        # Train
        save_features(net=net,
                      dataloader=train_loader,
                      data_name="train",
                      npy_save_dir=feature_save_path)

        # Valid
        save_features(net=net,
                      dataloader=valid_loader,
                      data_name="valid",
                      npy_save_dir=feature_save_path)

        # Test known
        save_features(net=net,
                      dataloader=test_known_known_loader_p0,
                      data_name="test_known_known_p0",
                      npy_save_dir=feature_save_path)

        save_features(net=net,
                      dataloader=test_known_known_loader_p1,
                      data_name="test_known_known_p1",
                      npy_save_dir=feature_save_path)

        save_features(net=net,
                      dataloader=test_known_known_loader_p2,
                      data_name="test_known_known_p2",
                      npy_save_dir=feature_save_path)

        save_features(net=net,
                      dataloader=test_known_known_loader_p3,
                      data_name="test_known_known_p3",
                      npy_save_dir=feature_save_path)

        # Test unknown
        save_features(net=net,
                      dataloader=test_unknown_unknown_loader,
                      data_name="test_unknown_unknown",
                      npy_save_dir=feature_save_path)

if __name__ == "__main__":
    main()
    
