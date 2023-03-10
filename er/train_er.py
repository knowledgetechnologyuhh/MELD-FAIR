import os
import copy
import argparse
import numpy as np
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix

from .data_utils import MELDDataset, collate_data
from .er_models import ERModel_BiModal, ERModel_Video, ERModel_Audio


def seed_everything():
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--result_folder_path', default = './results_av_er', help = 'Path of the folder to store the result logs and the trained model')
    parser.add_argument('--batch_size', type = int, default = 64, help = 'Batch size')
    parser.add_argument('--lr', type = float, default = 1e-4, help = 'Initial learning rate')
    parser.add_argument('--num_workers', type = int, default = 4, help = 'Number of threads')
    parser.add_argument('--video', action = 'store_true', default = True, help = 'Whether the ER model uses information from facial expressions')
    parser.add_argument('--audio', action = 'store_true', default = True, help = 'Whether the ER model uses information from speech')
    parser.add_argument('--face_sequence_length', type = int, default = 15, help = 'Number of consecutive facial expressions to be sent to the ER model')
    parser.add_argument('--sr', type = int, default = 16_000, help = 'Audio sample rate')
    parser.add_argument('--cuda_device_index', type = int, default = 0, help = 'Index of the GPU being used (-1 if running on the CPU)')

    return parser.parse_args()


def main(result_folder_path, batch_size = 64, initial_lr = 1e-4, num_workers = 4, video = True, audio = True, frame_sequence_length = 15, audio_sample_rate = 16_000, device_index = 0):
    MELD_folder = '../'
    splits = ['train', 'test', 'dev']
    classes = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    num_classes = len(classes)

    train_data = MELDDataset(MELD_folder, classes, 'train')
    test_data = MELDDataset(MELD_folder, classes, 'test')
    dev_data = MELDDataset(MELD_folder, classes, 'dev')
    
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = num_workers, collate_fn = lambda data : collate_data(data, 'train', num_classes, video, audio, frame_sequence_length, target_sr), pin_memory = True)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False, num_workers = num_workers, collate_fn = lambda data : collate_data(data, 'test', num_classes, video, audio, frame_sequence_length, target_sr), pin_memory = True)
    dev_loader = DataLoader(dev_data, batch_size = batch_size, shuffle = False, num_workers = num_workers, collate_fn = lambda data : collate_data(data, 'dev', num_classes, video, audio, frame_sequence_length, target_sr), pin_memory = True)
    
    softmax_layer = nn.Softmax(dim = 1)
    
    if torch.cuda.is_available() and device_index >= 0:
        device = torch.device(f'cuda:{device_index}')
    else:
        device = torch.device('cpu')
    
    if video:
        channel_mean = torch.Tensor([0.3067, 0.2233, 0.2040]).to(device)
        channel_std = torch.Tensor([0.1571, 0.1336, 0.1418]).to(device)
    
    if video and audio:
        model = ERModel_BiModal(num_classes, channel_mean, channel_std).to(device)
    elif video:
        model = ERModel_Video(num_classes, channel_mean, channel_std).to(device)
    elif audio:
        model = ERModel_Audio(num_classes).to(device)
    else:
        raise NotImplementedError
    
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    
    result_log = 'result_log.txt'
    history_log = 'history_log.txt'
    writer = SummaryWriter(os.path.join(result_folder_path, 'runs'))
    
    history = {}
    history['loss'] = {spl : [] for spl in splits}
    history['acc'] = {spl : [] for spl in splits}
    history['bal_acc'] = {spl : [] for spl in splits}
    history['f1'] = {spl : [] for spl in splits}
    
    last_epoch = -1
    
    best_epoch = last_epoch
    second_best_index = -1
    second_best_model = None
    best = {}
    best['model'] = model.state_dict()
    best['loss'] = 1e12
    best['acc'] = 0.0
    best['f1'] = 0.0
    best['bal_acc'] = 0.0
    best['cm'] = np.zeros((num_classes, num_classes))
    
    best_lr = initial_lr
    optimiser = optim.Adam(model.parameters(), lr = initial_lr)
    
    criterion = nn.CrossEntropyLoss()
    
    max_add_epochs = 30
    training_iterations = 0
    epoch = last_epoch
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size = 10, gamma = 0.5)
    
    while len(history['f1']['dev']) < 2 or epoch < history['f1']['dev'].index(best['f1']) + max_add_epochs or len(history['f1']['dev']) < max_add_epochs:
        if epoch > last_epoch + 1:
            scheduler.step()
        
        for spl in splits:        
            epochstart = datetime.now()
            print('\tEpoch {:d} - {} started at {:%Y-%m-%d %H:%M:%S}'.format(epoch + 1, 'Training' if spl == 'train' else 'Evaluation', epochstart))
            if result_folder_path is not None and result_log is not None:
                fout = open(os.path.join(result_folder_path, result_log), 'a')
                fout.write('\tEpoch {:d} - {} started at {:%Y-%m-%d %H:%M:%S}\n'.format(epoch + 1, 'Training' if spl == 'train' else 'Evaluation', epochstart))
                fout.close()
            
            label_lst = []
            pred_lst = []
            video_lst = []
            running_loss = 0.
            running_corrects = 0
            
            if spl == 'train':
                model.train()
                loader = train_loader
                torch.set_grad_enabled(True)
                optimiser.zero_grad()
            else:
                model.eval()
                if spl == 'dev':
                    loader = dev_loader
                elif spl == 'test':
                    loader = test_loader
                else:
                    raise NotImplementedError
                torch.set_grad_enabled(False)
            
            for idx_batch, batch in enumerate(loader):
                if video and audio:
                    face_inputs, audio_inputs, dialogues, utterances, labels = batch
                    face_inputs, audio_inputs, labels = face_inputs.to(device), audio_inputs.to(device).float(), labels.to(device)
                elif video:
                    face_inputs, dialogues, utterances, labels = batch
                    face_inputs, labels = face_inputs.to(device).float(), labels.to(device)
                elif audio:
                    audio_inputs, dialogues, utterances, labels = batch
                    audio_inputs, labels = audio_inputs.to(device).float(), labels.to(device)
                
                videos = torch.stack([dialogues, utterances])
                
                if video and audio:
                    outputs = model(face_inputs, audio_inputs)
                elif video:
                    outputs = model(face_inputs)
                elif audio:
                    outputs = model(audio_inputs)
                loss = criterion(outputs, labels)
                loss_value = loss.item()
                
                curr_label_lst = labels.detach().cpu().numpy().tolist()
                curr_pred_lst = outputs.detach().cpu().numpy().tolist()
                label_lst.extend(curr_label_lst)
                pred_lst.extend(curr_pred_lst)
                video_lst.extend([tuple(x) for x in videos.detach().cpu().numpy().tolist()])
                
                if spl == 'train':
                    loss.backward()
                    optimiser.step()
                    optimiser.zero_grad()
                    training_iterations += 1
                    
                    curr_pred_lst = torch.max(torch.Tensor(curr_pred_lst), dim = 1)[1].detach().cpu().numpy().tolist()
                    
                    accuracy = accuracy_score(curr_label_lst, curr_pred_lst)
                    bal_accuracy = balanced_accuracy_score(curr_label_lst, curr_pred_lst)
                    fscore = f1_score(curr_label_lst, curr_pred_lst, average = 'weighted')
                    
                    if batch_size > 1:
                        writer.add_scalar('Loss/{}/Batch'.format(spl.capitalize()), loss_value, training_iterations - 1)
                        writer.add_scalar('Accuracy/{}/Batch'.format(spl.capitalize()), round(accuracy, 4), training_iterations - 1)
                        writer.add_scalar('Balanced Accuracy/{}/Batch'.format(spl.capitalize()), round(bal_accuracy, 4), training_iterations - 1)
                        writer.add_scalar('Weighted F1-Score/{}/Batch'.format(spl.capitalize()), round(fscore, 4), training_iterations - 1)
                
                running_loss += loss_value
                
                if (idx_batch + 1) % 100 == 0:
                    currtime = datetime.now()
                    print('\t\t{:%Y-%m-%d %H:%M:%S} {:d}/{:d} batches processed.'.format(currtime, idx_batch + 1, len(loader) if spl != 'train' else len(train_loader_0) + len(train_loader_1)), end = '\r')
            
            currtime = datetime.now()
            print('\t\t{:%Y-%m-%d %H:%M:%S} {:d}/{:d} batches processed.'.format(currtime, len(loader), len(loader)))
            epoch_loss = running_loss / len(loader)
            torch.cuda.empty_cache()
            
            pred_lst = torch.max(torch.Tensor(pred_lst), dim = 1)[1].detach().cpu().numpy().tolist()
            
            accuracy = accuracy_score(label_lst, pred_lst)
            bal_accuracy = balanced_accuracy_score(label_lst, pred_lst)
            fscore = f1_score(label_lst, pred_lst, average = 'weighted')
            cm = confusion_matrix(label_lst, pred_lst)
            
            if epoch == last_epoch and spl != 'train' and len(history['loss'][spl]) > 0:
                if history['f1'][spl].index(best['f1']) == (last_epoch + 1):
                    best['model'] = second_best_model
                    best['loss'] = history['loss'][spl][second_best_index]
                    best['acc'] = history['acc'][spl][second_best_index]
                    best['bal_acc'] = history['bal_acc'][spl][second_best_index]
                    best['f1'] = history['f1'][spl][second_best_index]
                history['loss'][spl][-1] = epoch_loss
                history['acc'][spl][-1] = accuracy
                history['bal_acc'][spl][-1] = bal_accuracy
                history['f1'][spl][-1] = fscore
            else:
                history['loss'][spl].append(epoch_loss)
                history['acc'][spl].append(accuracy)
                history['bal_acc'][spl].append(bal_accuracy)
                history['f1'][spl].append(fscore)
            
            total_epochs = max_add_epochs + (0 if len(history['f1']['dev']) < 2 else history['f1']['dev'].index(best['f1']))
            print('{:%Y-%m-%d %H:%M:%S} {:^5} - [Epoch {:d}/{:d}] Loss: {:.4f} Acc: {:.2f} Bal Acc: {:.2f} F1: {:.2f}'.format(datetime.now(), spl.upper(),
                                                                                                                                epoch + 1, total_epochs,
                                                                                                                                history['loss'][spl][-1],
                                                                                                                                history['acc'][spl][-1],
                                                                                                                                history['bal_acc'][spl][-1],
                                                                                                                                history['f1'][spl][-1]))
            print(cm)
            
            writer.add_scalar('Loss/{}/Epoch'.format(spl.capitalize()), history['loss'][spl][-1], epoch + 1)
            writer.add_scalar('Accuracy/{}/Epoch'.format(spl.capitalize()), round(history['acc'][spl][-1], 4), epoch + 1)
            writer.add_scalar('Balanced Accuracy/{}/Epoch'.format(spl.capitalize()), round(history['bal_acc'][spl][-1], 4), epoch + 1)
            writer.add_scalar('Weighted F1-Score/{}/Epoch'.format(spl.capitalize()), round(history['f1'][spl][-1], 4), epoch + 1)
            
            if result_folder_path is not None and result_log is not None:
                fout = open(os.path.join(result_folder_path, result_log), 'a')
                fout.write('{:%Y-%m-%d %H:%M:%S} {:^5} - [Epoch {:d}/{:d}] Loss: {:.6f} Acc: {:.6f} Bal Acc: {:.6f} F1: {:.6f}\n'.format(datetime.now(), spl.upper(),
                                                                                                                                        epoch + 1, total_epochs,
                                                                                                                                        history['loss'][spl][-1],
                                                                                                                                        history['acc'][spl][-1],
                                                                                                                                        history['bal_acc'][spl][-1],
                                                                                                                                        history['f1'][spl][-1]))
                fout.write(str(cm) + '\n\n' + ('' if spl == 'train' else '\n'))
                fout.close()
            
            if spl == 'dev':
                if len(history['f1']['dev']) < 2 or history['f1']['dev'][-1] > best['f1']:
                    if len(history['f1']['dev']) < 2 or history['f1']['dev'][-1] > history['f1']['dev'][-2]:
                        if len(history['f1']['dev']) < 2:
                            second_best_index = -1
                        else:
                            second_best_model = copy.deepcopy(best['model'])
                            second_best_index = history['f1']['dev'].index(best['f1'])
                        best['model'] = copy.deepcopy(model.state_dict())
                        best['acc'] = history['acc']['dev'][-1]
                        best['bal_acc'] = history['bal_acc']['dev'][-1]
                        best['f1'] = history['f1']['dev'][-1]
                        best['loss'] = history['loss']['dev'][-1]
                        best['cm'] = np.copy(cm)
                    else:
                        second_best_index = history['f1']['dev'].index(best['f1'])
                        best['model'] = last_model_state
                        best['acc'] = history['acc']['dev'][-2]
                        best['bal_acc'] = history['bal_acc']['dev'][-2]
                        best['f1'] = history['f1']['dev'][-2]
                        best['loss'] = history['loss']['dev'][-2]
                        best['cm'] = last_cm
                    best_epoch = epoch
                    best_lr = optimiser.param_groups[0]['lr']
                    
                    # Save network
                    if result_folder_path is not None:
                        torch.save(model.state_dict(), os.path.join(result_folder_path, 'ER_Model.pth'))
                        print('Model saved at {}.'.format(os.path.join(result_folder_path, 'ER_Model.pth')))
                
                last_model_state = copy.deepcopy(model.state_dict())
                last_cm = np.copy(cm)
        
        epoch += 1


if __name__ == '__main__':
    seed_everything()
    args = parse_arguments()
    main(args.result_folder_path, args.batch_size, args.lr, args.num_workers, args.video, args.audio, args.face_sequence_length, args.sr, args.cuda_device_index)
