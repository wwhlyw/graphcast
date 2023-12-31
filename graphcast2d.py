from graphconstruct import icosahedral_mesh, grid_mesh_connectivity
from graphconstruct import features
from data.dataset import HRRR
from model.graphcastnet import GraphCastNet
from model.loss import Loss
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import LambdaLR
import math
import torch.nn as nn
import os


class GraphCast:
    def __init__(self, config):
        self.model_config = config['model']
        self.data_config = config['data']
        self.train_config = config['train']

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.variables_list = ["z50", "z500", "z850", "z1000", "t50", "t500", "t850", "t1000", 
                               "s50", "s500", "s850", "s1000", "u50", "u500", "u850", "u1000", 
                               "v50", "v500", "v850", "v1000", "mslp", "u10", "v10", "t2m",] 

        # sj, wj, ai
        self.sj = torch.from_numpy(np.load(self.data_config['sj'])).to(self.device)
        self.wj = torch.from_numpy(np.load(self.data_config['wj'])).to(self.device)
        # self.ai = np.load(self.data_config['ai'])

        # meshes list initialization
        init_mesh = icosahedral_mesh.get_quadrangle()
        self._meshes = (icosahedral_mesh.meshes_list(splits=self.model_config['splits'], current_mesh=init_mesh))
        
        self._init_properties()

    def train(self):
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_criterion()

        print('-------------start training-------------')
        print('--------------train phase1--------------')
        scheduler = self._get_scheduler(1)
        self.idx = 1
        for epoch in range(self.train_config['phase1_epoch']):
            train_loader = self._init_data(mode='train')
            train_loss = self.train_one_epoch_1or2_phase(train_loader, epoch)
            scheduler.step()
            print(f'epoch: {train_loss}')
            self.write_to_file(folder_path='/home/wwh/graphcast/train', file_name='train.txt', content=f'epoch:{train_loss}\n') 
            self.valid()         
        
        print('--------------train phase2--------------')
        scheduler = self._get_scheduler(2)    
        for epoch in range(self.train_config['phase1_epoch'], self.train_config['phase2_epoch']):
            train_loader = self._init_data(mode='train')
            train_loss = self.train_one_epoch_1or2_phase(train_loader, epoch)
            scheduler.step()
            print(f'epoch: {train_loss}')
            self.write_to_file(folder_path='/home/wwh/graphcast/train', file_name='train.txt', content=f'epoch:{train_loss}\n') 
            self.valid()
        
        print('--------------train phase3--------------')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-7, betas=[0.9, 0.95], weight_decay=0.1)
        scheduler = self._get_scheduler(3)
        output_timestamps = 2
        reset_steps = self.train_config['reset_steps']
        for epoch in range(self.train_config['phase2_epoch'], self.train_config['phase3_epoch']):
            train_loader = self._init_data(mode='train', output_timestamps=output_timestamps)
            train_loss = self.train_one_epoch_3_phase(train_loader, output_timestamps, epoch)
            print(f'epoch: {train_loss}')
            self.valid(steps=output_timestamps) 
            if epoch % reset_steps == 0:
                output_timestamps += 1 
            if output_timestamps > self.model_config['output_timestamps']:
                output_timestamps = self.model_config['output_timestamps']
                path = f'{self.model_config["save_path"]}/{epoch}.pth'
                torch.save(self.model.state_dict(), path)
            

    def train_one_epoch_1or2_phase(self, train_loader, epoch): 
        train_loss = []
        for i, (input, label, input_forcings, label_forcings) in enumerate(train_loader):
            
            self.optimizer.zero_grad()
            
            input = input.to(self.device)
            label = label.to(self.device)
            input_forcings1 = torch.unsqueeze(input_forcings, dim=1).expand([-1, input.shape[1], -1]).to(self.device)
            input_forcings2 = torch.unsqueeze(label_forcings, dim=1).expand([-1, input.shape[1], -1]).to(self.device)
            input_forcings = torch.concat([input_forcings1, input_forcings2], dim=-1)
            constant = torch.stack([np.cos(self.grid_lat), 
                                    np.cos(self.grid_lon),
                                    np.sin(self.grid_lon)], dim=1).unsqueeze(dim=0).expand([input.shape[0], -1, -1]).to(self.device)
        
            # input.shape[batch, num_grid, features] 
            # [32, 179520, 57]
            # features:[v_{t-1}, v{t}, f_{t-1}, f_{t}, f_{t+1}, constant] [24+24+2+2+2+3]
            input = torch.concat([input, input_forcings, constant], axis=-1).to(self.device).to(torch.float32)
            
            predict = self.model(input)
            label = label * self.per_variable_level_std + self.per_variable_level_mean
            loss = self.criterion(predict, label)
            
            train_loss.append(loss.item())
            if (i+1) % 100 == 0:
                print(f'\t epoch: {epoch}|iter: {i}: train_loss: {loss.item()}')
            loss.backward()

            max_norm = 32
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            self.optimizer.step()

        return torch.mean(torch.tensor(train_loss))

    def train_one_epoch_3_phase(self, train_loader, steps, epoch):
        train_loss = []
        vars = self.model_config['variables']
        self.criterion = self._get_criterion(steps)
        for i, (input, label, input_forcings, label_forcings) in enumerate(train_loader):
            self.optimizer.zero_grad()
            input = input.to(self.device)
            label = label.to(self.device)
            label_forcings = label_forcings.to(self.device)
            predicts = []
            for j in range(steps):
                input_forcings_new1 = input_forcings.to(self.device)
                input_forcings_new2 = label_forcings[:, 2*j:2*(j+1)].to(self.device)
                input_forcings = torch.cat([input_forcings_new1, input_forcings_new2], dim=-1).to(self.device)
                input_forcings_new = torch.unsqueeze(input_forcings, dim=1).expand([-1, input.shape[1], -1])
                constant = torch.stack([np.cos(self.grid_lat), 
                                        np.cos(self.grid_lon),
                                        np.sin(self.grid_lon)], dim=1).unsqueeze(dim=0).expand([input.shape[0], -1, -1]).to(self.device)
              
                input = torch.concat([input, input_forcings_new, constant], axis=-1).to(self.device).float()
            
                predict = self.model(input)
                
                predicts.append(predict)

                input = torch.cat([input[:,:, vars:2*vars], predict], axis=-1)
                input_forcings = input_forcings[:, 2:]
            
            predict_all = torch.cat(predicts, axis=-1)

            per_variable_level_mean = self.per_variable_level_mean.unsqueeze(0).unsqueeze(0).repeat([1, 1, steps])
            per_variable_level_std = self.per_variable_level_std.unsqueeze(0).unsqueeze(0).repeat([1, 1, steps])
            label = label * per_variable_level_std + per_variable_level_mean

            loss = self.criterion(predict_all, label, steps)
            train_loss.append(loss.item())

            if i % 100 == 0:
                print(f'\t epoch: {epoch}|iter: {i}: train_loss: {loss.item() / input.shape[1]}')
            loss.backward()
            max_norm = 32
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            self.optimizer.step()
    

        return torch.mean(torch.tensor(train_loss)) 

    def valid(self, steps=1):
        valid_loader = self._init_data('valid', output_timestamps=steps)
        vars = self.model_config['variables']
        self.model.eval()
        predict_all = []
        target_all = []
        with torch.no_grad():
            for i, (input, label, input_forcings, label_forcings) in enumerate(valid_loader):
                input = input.to(self.device)
                label = label.to(self.device)
                label_forcings = label_forcings.to(self.device)
                predicts = []
                for j in range(steps):
                    input_forcings_new1 = input_forcings.to(self.device)
                    input_forcings_new2 = label_forcings[:, 2*j:2*(j+1)].to(self.device)
                   
                    input_forcings = torch.cat([input_forcings_new1, input_forcings_new2], dim=-1).to(self.device)
                    input_forcings_new = torch.unsqueeze(input_forcings, dim=1).expand([-1, input.shape[1], -1])
                    
                    constant = torch.stack([np.cos(self.grid_lat), 
                                            np.cos(self.grid_lon),
                                            np.sin(self.grid_lon)], dim=1).unsqueeze(dim=0).expand([input.shape[0], -1, -1]).to(self.device)
                    
                
                    input = torch.concat([input, input_forcings_new, constant], axis=-1).float()
                
                    predict = self.model(input)
                    
                    predicts.append(predict)

                    input = torch.cat([input[:, :, vars:2*vars], predict], axis=-1)
                    input_forcings = input_forcings[:, 2:]
                
                per_variable_level_mean = self.per_variable_level_mean.unsqueeze(0).unsqueeze(0).repeat([1, 1, steps])
                per_variable_level_std = self.per_variable_level_std.unsqueeze(0).unsqueeze(0).repeat([1, 1, steps])
                label = label * per_variable_level_std + per_variable_level_mean
                # [B, N, F * T]
                tmp = torch.cat(predicts, axis=-1)
                predict_all.append(tmp)
                target_all.append(label)
            # [B, N, F * T] -> [F * T, B, N]

            predict_all = torch.cat(predict_all, dim=0).permute(2, 0, 1)
            target_all = torch.cat(target_all, dim=0).permute(2, 0, 1)
            for i in range(steps):
                for j, variable in enumerate(self.variables_list):
                    folder_path = '/home/wwh/graphcast/predict/' + str(i)
                    file_name = variable + '.txt'
                    RMSE = torch.sqrt(torch.mean((predict_all[i*24+j] - target_all[i*24+j]) ** 2))
                    content = f'rmse:{RMSE}\n'
                    self.write_to_file(folder_path, file_name, content)
            RMSE = torch.sqrt(torch.mean((predict_all-target_all) ** 2))
            folder_path = '/home/wwh/graphcast/predict/' + str(i)
            file_name = 'total.txt'
            content = f'rmse:{RMSE}\n'
            self.write_to_file(folder_path, file_name, content)
            print(f'VALID RMSE: {RMSE}')

    def _init_properties(self):
        self._init_mesh_properties()
        self._init_grid_properties()
        self._init_grid2mesh_graph()
        self._init_mesh2mesh_graph()
        self._init_mesh2grid_graph()
        self._init_model()
        print('----------------init---------------------')                                                  


    def _init_mesh_properties(self):
        self.mesh_pos = self._finest_mesh.vertices[:, :2]

    
    def _init_grid_properties(self):
        """
        input:
        grid_lat: [num_lat, num_lon] latitude of grid nodes
        grid_lon: [num_lat, num_lon] longitude of grid nodes     
        """
        grid_y = torch.arange(408).repeat(440) / 440.
        grid_x = torch.arange(440).reshape([-1, 1]).repeat([-1, 408]).reshape([-1]) / 440.
        # self.grid_lon: [num_grid]  self.grid_lat: [num_grid]  1维形式 
        self.grid_x = torch.tensor(grid_x, dtype=torch.float32)
        self.grid_y = torch.tensor(grid_y, dtype=torch.float32)
        senders_pos = []
        for i in range(408):
            for j in range(440):
                senders_pos.append([i / 440., j / 440.])
        # 2维形式
        self.grid_pos = torch.tensor(senders_pos, dtype=torch.float32)
 
    def _init_grid2mesh_graph(self):
        self.g2m_src_idx, self.g2m_dst_idx = grid_mesh_connectivity.g2m_or_m2g_edges_indices_2d(
            mesh=self._finest_mesh,
            radius=self.model_config['radius'],
            type='g2m'
        )
        
        senders_node_features, receivers_node_features, edge_features = features.get_bipartite_graph_spatial_features_2d(
            senders_pos = self.grid_pos,
            receivers_pos= self.mesh_pos,
            senders = self.g2m_src_idx,
            receivers = self.g2m_dst_idx,
        )

        self.grid_constant_feats = torch.tensor(senders_node_features, dtype=torch.float32).to(self.device)
        self.mesh_node_feats = torch.tensor(receivers_node_features, dtype=torch.float32).to(self.device)
        self.g2m_edge_feats = torch.tensor(edge_features, dtype=torch.float32).to(self.device)
      
        self.g2m_src_idx = torch.from_numpy(self.g2m_src_idx).to(self.device)
        self.g2m_dst_idx = torch.from_numpy(self.g2m_dst_idx).to(self.device)

    
    def _init_mesh2mesh_graph(self):
        merged_mesh = icosahedral_mesh.merge_meshes(self._meshes)
        senders, receivers = grid_mesh_connectivity.mesh2mesh_edge_indices_2d(merged_mesh.faces)
        edge_features = features.get_homogeneous_graph_spatial_features(
            node_lat = self.mesh_pos,
            node_lon = self.mesh_pos,
            senders = senders,
            receivers = receivers,
        )
        clip_senders = []
        clip_receivers = []
        clip_edge_index = []
        mesh_connects = set(self.g2m_dst_idx.tolist())
        raw2clip = {mesh_idx:i for i, mesh_idx in enumerate(mesh_connects)} 
        for i, (sender, receiver) in enumerate(zip(senders, receivers)):
            if sender in mesh_connects and receiver in mesh_connects:
                clip_senders.append(raw2clip[sender])
                clip_receivers.append(raw2clip[receiver])
                clip_edge_index.append(i)
        
        self.mesh_connects = torch.tensor(list(mesh_connects), dtype=torch.int64).to(self.device)
        self.clip_edge_index = torch.tensor(clip_edge_index, dtype=torch.int64).to(self.device)
        self.mesh_edge_feats = torch.tensor(edge_features, dtype=torch.float32).to(self.device)
        self.m2m_src_idx = torch.tensor(clip_senders, dtype=torch.int64).to(self.device)
        self.m2m_dst_idx = torch.tensor(clip_receivers, dtype=torch.int64).to(self.device)

    def _init_mesh2grid_graph(self):
        grid_indices, mesh_indices = grid_mesh_connectivity.g2m_or_m2g_edges_indices_2d(
            mesh=self._finest_mesh,
            radius=self.model_config['radius'],
            type='m2g'
        )

        senders = mesh_indices,
        receivers = grid_indices,

        _, _, edge_features = features.get_bipartite_graph_spatial_features_2d(
            senders_pos = self.mesh_pos,
            receivers_pos= self.grid_pos,
            senders=senders,
            receivers=receivers,
        )

        self.m2g_edge_feats = torch.tensor(edge_features, dtype=torch.float32).to(self.device)
        self.m2g_src_idx = torch.tensor(mesh_indices, dtype=torch.int64).to(self.device)
        self.m2g_dst_idx = torch.tensor(grid_indices, dtype=torch.int64).to(self.device)

    def _init_model(self):
        self.per_variable_level_mean = torch.from_numpy(np.load(self.data_config['mean'])).to(self.device)
        self.per_variable_level_std = torch.from_numpy(np.load(self.data_config['std'])).to(self.device)
        B = self.data_config['batch_size']
        self.mesh_node_feats = self.mesh_node_feats.unsqueeze(0).expand([B, -1, -1])
        self.mesh_edge_feats = self.mesh_edge_feats.unsqueeze(0).expand([B, -1, -1])
        self.g2m_edge_feats = self.g2m_edge_feats.unsqueeze(0).expand([B, -1, -1])
        self.m2g_edge_feats = self.m2g_edge_feats.unsqueeze(0).expand([B, -1, -1])

        input_channels = self.model_config['variables'] * self.data_config['input_timestamps'] \
                        + self.model_config['forcings'] * (self.data_config['input_timestamps'] + 1) \
                        + self.model_config['constant']
        self.model = GraphCastNet(
            vg_in_channels=input_channels,
            vg_out_channels=self.model_config['variables'],
            vm_in_channels=self.model_config['vm_in_channels'],
            em_in_channels=self.model_config['em_in_channels'],
            eg2m_in_channels=self.model_config['eg2m_in_channels'],
            em2g_in_channels=self.model_config['em2g_in_channels'],
            latent_dims=self.model_config['latent_dims'],
            processing_steps=self.model_config['processing_steps'],
            g2m_src_idx=self.g2m_src_idx,
            g2m_dst_idx=self.g2m_dst_idx,
            m2m_src_idx=self.m2m_src_idx,
            m2m_dst_idx=self.m2m_dst_idx,
            m2g_src_idx=self.m2g_src_idx,
            m2g_dst_idx=self.m2g_dst_idx,
            mesh_node_feats=self.mesh_node_feats,
            mesh_edge_feats=self.mesh_edge_feats,
            g2m_edge_feats=self.g2m_edge_feats,
            m2g_edge_feats=self.m2g_edge_feats,
            clip_edge_idx=self.clip_edge_index,
            mesh_connects=self.mesh_connects,
            per_variable_level_mean=self.per_variable_level_mean,
            per_variable_level_std=self.per_variable_level_std
        ).to(self.device)


    def _init_data(self, mode, output_timestamps=1):
        if mode == 'train':
            hrrr_data = HRRR(self.data_config[mode], self.data_config['input_timestamps'], output_timestamps)
            data_loader = DataLoader(hrrr_data, batch_size=self.data_config['batch_size'], shuffle=True)                
        else:
            hrrr_data = HRRR(self.data_config[mode], self.data_config['input_timestamps'], output_timestamps)
            data_loader = DataLoader(hrrr_data, batch_size=self.data_config['batch_size'])
        return data_loader

    def _get_criterion(self, steps=1):
        criterion = Loss(self.sj, self.wj, steps)
        return criterion 

    def _get_optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=0.001, betas=[0.9, 0.95], weight_decay=0.1)       
    
    def _get_scheduler(self, phase):
        if phase == 1:
            def lr_lambda1(epoch):
                initial_lr = 0
                final_lr = 0.001
                # num_epochs = self.train_config['phase1_epoch']
                num_epochs = 1
                return initial_lr + (final_lr - initial_lr) * (epoch / num_epochs)
            return LambdaLR(self.optimizer, lr_lambda=lr_lambda1)
        elif phase == 2:
            def lr_lambda2(epoch):
                initial_lr = 0.001
                final_lr = 0
                total_steps = self.train_config['phase2_epoch'] - self.train_config['phase1_epoch']
                return final_lr + 0.5 * (initial_lr - final_lr) * (1 + math.cos(math.pi * epoch / total_steps))
            return LambdaLR(self.optimizer, lr_lambda=lr_lambda2)
        else:
            return LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)
    
    def write_to_file(self, folder_path, file_name, content):
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'a') as file:
                file.write(content)
        except FileNotFoundError:
            if not os.path.exists(folder_path): 
                os.makedirs(folder_path)
            with open(file_path, 'w') as file:
                file.write(content)

    @property
    def _finest_mesh(self):
        return self._meshes[-1]