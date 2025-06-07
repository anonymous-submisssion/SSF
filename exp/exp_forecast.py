import os
import time
import warnings
import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from utils.metrics import metric
from models import SSF



warnings.filterwarnings('ignore')


class Forecast(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'SSF': SSF,
        }

        dataset_name = self.args.data_name
        horizon = self.args.horizon

        num_sensors = {'METR-LA':207, 
                    'PEMS-BAY':325,
                    'NAVER-Seoul':774,
                    'PEMS04':307,
                    'PEMS08':170}

        data_dim = {'METR-LA':2, 
                    'PEMS-BAY':2,
                    'NAVER-Seoul':2,
                    'PEMS04':9,
                    'PEMS08':9}


        self.args.num_sensors = num_sensors[dataset_name]
        self.args.data_dim = data_dim[dataset_name]
        
        self.dataset_dir = os.path.join("Datasets", f"horizon_{horizon}", dataset_name)

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def inverse_transform(self, scaler, model_outputs):
        """
        Apply inverse transform of the scaler to model outputs
        
        Args:
            scaler: StandardScaler instance used for normalization
            model_outputs: PyTorch tensor of model predictions (normalized)
            
        Returns:
            torch.Tensor: Denormalized outputs
        """
        # Make sure model_outputs is on CPU and convert to numpy
        if isinstance(model_outputs, torch.Tensor):
            if model_outputs.is_cuda:
                model_outputs = model_outputs.cpu()
            # Handle various tensor shapes
            original_shape = model_outputs.shape
            
            normalized_channel = model_outputs[..., 0].detach().numpy()
            
            # Reshape for inverse transform
            normalized_channel_flat = normalized_channel.reshape(-1, 1)
            denormalized_channel_flat = scaler.inverse_transform(normalized_channel_flat)
            
            # Reshape back to original shape
            denormalized_channel = denormalized_channel_flat.reshape(normalized_channel.shape)
            
            result = model_outputs.clone()
            result[..., 0] = torch.from_numpy(denormalized_channel).float()
            
            return result.to(self.device)
        else:
            raise TypeError("Expected PyTorch tensor, got {}".format(type(model_outputs)))


    def _get_data(self, flag):

        data = {}
        batch_size = self.args.batch_size
        # Load data for train, val, test
        for category in ['train', 'val', 'test']:
            cat_data = np.load(os.path.join(self.dataset_dir , category + '.npz'))
            data['x_' + category] = torch.tensor(cat_data['x'], dtype=torch.float32)
            data['y_' + category] = torch.tensor(cat_data['y'], dtype=torch.float32)
        
        scaler = StandardScaler()
        scaler.fit(data['x_train'][..., 0].numpy().reshape(-1, 1))
        
        # Transform data using scaler
        
        for category in ['train', 'val', 'test']:
            # Transform first channel (assuming it's the channel that needs scaling)
            data['x_' + category][..., 0] = torch.tensor(
                scaler.transform(data['x_' + category][..., 0].numpy().reshape(-1, 1)).reshape(data['x_' + category][..., 0].shape), 
                dtype=torch.float32
            )
            if 'METR' in self.args.data_name or 'BAY' in self.args.data_name:
                data['y_' + category][..., 0] = data['y_' + category].mean() + torch.tensor(
                    scaler.transform(data['y_' + category][..., 0].numpy().reshape(-1, 1)).reshape(data['y_' + category][..., 0].shape), 
                    dtype=torch.float32
                )
            
        # Create TensorDatasets and DataLoaders
        data['train_loader'] = DataLoader(
            TensorDataset(data['x_train'], data['y_train']), 
            batch_size=batch_size, 
            shuffle=True
        )
        data['val_loader'] = DataLoader(
            TensorDataset(data['x_val'], data['y_val']), 
            batch_size=batch_size, 
            shuffle=False
        )
        data['test_loader'] = DataLoader(
            TensorDataset(data['x_test'], data['y_test']), 
            batch_size=batch_size, 
            shuffle=False
        )
        data['scaler'] = scaler
        print("Successfully loaded the Spatio-temporal {} Dataset!".format(flag))
        
        if flag == 'train':
            return scaler, data['train_loader'] 
        elif flag == 'val':
            return scaler, data['val_loader'] 
        else:
            return scaler, data['test_loader'] 

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, eps=1e-7, weight_decay=0.5)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, scaler):
        total_loss = []
        vali_losses = []  # Store batch-wise validation loss

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):

                batch_x = batch_x.float().to(self.device)
                batch_x = batch_x.reshape(batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]*batch_x.shape[3])
                batch_y = batch_y.reshape(batch_y.shape[0], batch_y.shape[1], batch_y.shape[2]*batch_y.shape[3])
                if batch_x.shape[0] < self.args.batch_size:
                    continue
                B, T, N = batch_x.shape
                outputs = self.model(batch_x)

                batch_y = batch_y.to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
                vali_losses.append(loss.item())  # Store loss for each batch

        total_loss = np.average(total_loss)
        self.model.train()

        return total_loss

    def train(self, setting):
        scaler, train_loader = self._get_data(flag='train')
        scaler, vali_loader = self._get_data(flag='val')
        scaler, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_epoch_losses = []  # Store epoch-wise training loss
        valid_epoch_losses = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x = batch_x.reshape(batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]*batch_x.shape[3])
                batch_y = batch_y.reshape(batch_y.shape[0], batch_y.shape[1], batch_y.shape[2]*batch_y.shape[3])
                if batch_x.shape[0] < self.args.batch_size:
                    continue
                # random mask
                B, T, N = batch_x.shape

                outputs = self.model(batch_x)
                
                batch_y = batch_y.to(self.device)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            train_epoch_losses.append(train_loss)  # Store loss for the current epoch

            vali_loss = self.vali(torch.Tensor([0]), vali_loader, criterion, scaler)
            valid_epoch_losses.append(vali_loss)
            test_loss = self.vali(torch.Tensor([0]), test_loader, criterion, scaler)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # Get current date and time as a string
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        return self.model

    def test(self, setting, test=0):
        scaler, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x = batch_x.reshape(batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]*batch_x.shape[3])
                batch_y = batch_y.reshape(batch_y.shape[0], batch_y.shape[1], batch_y.shape[2]*batch_y.shape[3])
                # random mask
                if batch_x.shape[0] < self.args.batch_size:
                    continue
                B, T, N = batch_x.shape

                # imputation
                outputs = self.model(batch_x)

                #print("out : ", outputs)
                #print("target : ", batch_y)
                # eval
                batch_y = batch_y.to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                #print("out: ", outputs)
                #print("batch_y: ", batch_y)

                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{}'.format(mse, mae, rmse, mape))
        f = open("result_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}'.format(mse, mae, rmse, mape))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)


        return

def plot_and_save_matrices(preds, trues, filename=''):

    preds = np.abs(preds.T)
    trues = np.abs(trues.T)
    plt.figure(figsize=(10, 5))
    plt.plot(trues[2][1000:3000], label='True Values', linewidth=0.7, color='purple')
    plt.plot(preds[2][1000:3000], label='Predicted Values', linewidth=0.7, color='darkorange')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    



class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
