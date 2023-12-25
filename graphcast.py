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


class GraphCast:
    def __init__(self, config):
        self.model_config = config['model']
        self.data_config = config['data']
        self.train_config = config['train']

        # data path
        self.lon_path = self.data_config['lon_path']
        self.lat_path = self.data_config['lat_path']

        self.raw_grid_lat = np.load(self.lat_path)[253:693,970:1378]
        self.raw_grid_lon = np.load(self.lon_path)[253:693,970:1378]

        # sj, wj, ai
        self.sj = np.load(self.data_config['sj'])
        self.wj = np.load(self.data_config['wj'])
        self.ai = np.load(self.data_config['ai'])

        # meshes list initialization
        self._meshes = (icosahedral_mesh.meshes_list(splits=self.model_config['splits']))
        
        self._init_properties(self.raw_grid_lat, self.raw_grid_lon)

    def train(self):
        self.grid_lat, self.grid_lon = torch.tensor(self.grid_lat), torch.tensor(self.grid_lon)
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_criterion()
        scheduler = self._get_scheduler(1)
        for epoch in range(self.train_config['phase1_epoch']):
            train_loader = self._init_data(mode='train')
            train_loss = self.train_one_epoch_1or2_phase(train_loader, epoch)
            scheduler.step()
            print(f'epoch: {train_loss}')          
        scheduler = self._get_criterion(2)    
        for epoch in range(self.train_config['phase1_epoch'], self.train_config['phase2_epoch']):
            train_loader = self._init_data(mode='train')
            train_loss = self.train_one_epoch_1or2_phase(train_loader, epoch)
            scheduler.step()
            print(f'epoch: {train_loss}')
        scheduler = self._get_criterion(3)
        output_timestamps = 2
        reset_steps = self.model_config['reset_steps']
        for epoch in range(self.train_config['phase2_epoch'], self.train_config['phase3_epoch']):
            train_loader = self._init_data(mode='train', output_timestamps=output_timestamps)
            train_loss = self.train_one_epoch_3_phase(train_loader, output_timestamps, epoch)
            print(f'epoch: {train_loss}')
            if epoch % reset_steps == 0:
                output_timestamps += 1 
            if output_timestamps > self.model_config['output_timestamps']:
                output_timestamps = self.model_config['output_timestamps']  

    def train_one_epoch_1or2_phase(self, train_loader, epoch): 
        train_loss = []
        for i, (input, label, input_forcings, label_forcings) in enumerate(train_loader):
            self.optimizer.zero_grad()
            input_forcings = torch.unsqueeze(input_forcings, dim=1).expand([-1, input.shape[1], -1])
            constant = torch.stack([self.grid_lat, self.grid_lon], dim=1).unsqueeze(dim=0).expand([input.shape[0], -1, -1])
            input = torch.concat([input, input_forcings, constant], axis=-1)
            predict = self.model(input)
            loss = self.criterion(predict, label)
            train_loss.append(loss.item())
            if i % 100 == 0:
                print(f'\t epoch: {epoch}|iter: {i}: train_loss: {loss.item()}')
            loss.backward()
            self.optimizer.step()

        return torch.mean(train_loss)

    def train_one_epoch_3_phase(self, train_loader, steps, epoch):
        train_loss = []
        vars = self.model_config['variables']
        for i, (input, label, input_forcings, label_forcings) in enumerate(train_loader):
            self.optimizer.zero_grad()
            predicts = []
            for j in range(steps):
                input_forcings = torch.unsqueeze(input_forcings, dim=1).expand([-1, input.shape[1], -1])
                constant = torch.stack([self.grid_lat, self.grid_lon], dim=1).unsqueeze(dim=0).expand([input.shape[0], -1, -1])
                input = torch.concat([input, input_forcings, constant], axis=-1)
                predict = self.model(input)
                
                predicts.append(predict)

                input = torch.cat([input[:,:, vars:2*vars], predict], axis=-1)
                input_forcings = torch.cat([input_forcings[:, 2:], label_forcings[:, 2*j:2*(j+1)]], axis=-1)
            predict_all = torch.cat(predict, axis=-1)
            loss = self.criterion(predict_all, label, steps)
            train_loss.append(loss.item())
            if i % 100 == 0:
                print(f'\t epoch: {epoch}|iter: {i}: train_loss: {loss.item()}')
            loss.backward()
            self.optimizer.step()

        return torch.mean(train_loss) 

    def valid(self):
        out_timestamps = self.model_config['output_timestamps']
        valid_loader = self._init_data('valid')
        self.model.eval()
        valid_loss = []
        with torch.no_grad():
            for i, (input, label, input_forcings, label_forcings) in enumerate(valid_loader):
                predicts = []
                for j in range(out_timestamps):
                    input_forcings = torch.unsqueeze(input_forcings, dim=1).expand([-1, input.shape[1], -1])
                    constant = torch.stack([self.grid_lat, self.grid_lon], dim=1).unsqueeze(dim=0).expand([input.shape[0], -1, -1])
                    input = torch.concat([input, input_forcings, constant], axis=-1)
                    predict = self.model(input)
                    
                    predicts.append(predict)

                    input = torch.cat([input[:,:, vars:2*vars], predict], axis=-1)
                    input_forcings = torch.cat([input_forcings[:, 2:], label_forcings[:, 2*j:2*(j+1)]], axis=-1)
            predict_all = torch.cat(predict, axis=1)
            loss = nn.MSELoss(predict_all, label)
            


    def _init_properties(self, grid_lat, grid_lon):
        self._init_mesh_properties()
        self._init_grid_properties(grid_lat, grid_lon)
        self._init_grid2mesh_graph()
        self._init_mesh2mesh_graph()
        self._init_mesh2grid_graph()
        self._init_model()                                                  


    def _init_mesh_properties(self):
        self._num_mesh_nodes = self._finest_mesh.vertices.shape[0]
        mesh_phi, mesh_theta = features.cartesian_to_spherical(
            self._finest_mesh.vertices[:, 0],
            self._finest_mesh.vertices[:, 1],
            self._finest_mesh.vertices[:, 2],
        )
        mesh_lat, mesh_lon = features.spherical_to_lat_lon(
            phi=mesh_phi, theta=mesh_theta
        )

        self.mesh_lat = mesh_lat.astype(np.float32)
        self.mesh_lon = mesh_lon.astype(np.float32)
    
    def _init_grid_properties(self, grid_lat, grid_lon):
        """
        input:
        grid_lat: [num_lat, num_lon] latitude of grid nodes
        grid_lon: [num_lat, num_lon] longitude of grid nodes     
        """
        self._num_grid_nodes = grid_lat.shape[0] * grid_lat.shape[1]

        # self.grid_lon: [num_grid]  self.grid_lat: [num_grid]   
        self.grid_lon = grid_lon.reshape([-1]).astype(np.float32)
        self.grid_lat = grid_lat.reshape([-1]).astype(np.float32)
 
    def _init_grid2mesh_graph(self):
        self.g2m_src_idx, self.g2m_dst_idx = grid_mesh_connectivity.grid2mesh_edges_indices(
            grid_latitude=self.raw_grid_lat,
            grid_longitude=self.raw_grid_lon,
            mesh=self._finest_mesh,
            radius=self.model_config['radius']
        )
        
        senders_node_features, receivers_node_features, edge_features = features.get_bipartite_graph_spatial_features(
            senders_lat = self.grid_lat,
            senders_lon = self.grid_lon,
            receivers_lat = self.mesh_lat,
            receivers_lon = self.mesh_lon,
            senders = self.g2m_src_idx,
            receivers = self.g2m_dst_idx,
        )
        self.grid_constant_feats = senders_node_features
        self.mesh_node_feats = receivers_node_features
        self.g2m_edge_feats = edge_features
    
    def _init_mesh2mesh_graph(self):
        merged_mesh = icosahedral_mesh.merge_meshes(self._meshes)
        senders, receivers = grid_mesh_connectivity.mesh2mesh_edge_indices(merged_mesh.faces)
        edge_features = features.get_homogeneous_graph_spatial_features(
            node_lat = self.mesh_lat,
            node_lon = self.mesh_lon,
            senders = senders,
            receivers = receivers,
        )
        self.mesh_edge_feats = edge_features
        self.m2m_src_idx=senders
        self.m2m_dst_idx=receivers

    def _init_mesh2grid_graph(self):
        grid_indices, mesh_indices = grid_mesh_connectivity.mesh2grid_edge_indices(
            grid_latitude=self.raw_grid_lat,
            grid_longitude=self.raw_grid_lon,
            mesh=self._finest_mesh,
        )

        senders = mesh_indices,
        receivers = grid_indices,

        _, _, edge_features = features.get_bipartite_graph_spatial_features(
            senders_lat=self.mesh_lat,
            senders_lon=self.mesh_lon,
            receivers_lat=self.grid_lat,
            receivers_lon=self.grid_lon,
            senders=senders,
            receivers=receivers,
        )
        self.m2g_edge_feats = edge_features
        self.m2g_src_idx = mesh_indices
        self.m2g_dst_idx = grid_indices

    def _init_model(self):
        self.per_variable_level_mean = np.load(self.data_config['mean'])
        self.per_variable_level_std = np.load(self.data_config['std'])

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
            per_variable_level_mean=self.per_variable_level_mean,
            per_variable_level_std=self.per_variable_level_std
        )

    def _init_data(self, mode, output_timestamps=1):
        if mode == 'train':
            hrrr_data = HRRR(self.data_config[mode], self.data_config['input_timestamps'], output_timestamps)
            data_loader = DataLoader(hrrr_data, batch_size=self.data_config['batch_size'], shuffle=True)                
        else:
            hrrr_data = HRRR(self.data_config[mode], self.data_config['input_timestamps'], output_timestamps)
            data_loader = DataLoader(hrrr_data, batch_size=self.data_config['batch_size'])
        return data_loader

    def _get_criterion(self):
        criterion = Loss(self.sj, self.wj, self.ai)
        return criterion 

    def _get_optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=0.001, betas=[0.9, 0.95], weight_decay=0.1)       
    
    def _get_scheduler(self, phase):
        if phase == 1:
            def lr_lambda1(epoch):
                initial_lr = 0
                final_lr = 0.001
                num_epochs = 1000
                return initial_lr + (final_lr - initial_lr) * (epoch / num_epochs)
            return LambdaLR(self.optimizer, lr_lambda=lr_lambda1)
        elif phase == 2:
            def lr_lambda2(epoch):
                initial_lr = 0.001
                final_lr = 0
                total_steps = 300000
                return final_lr + 0.5 * (initial_lr - final_lr) * (1 + math.cos(math.pi * epoch / total_steps))
            return LambdaLR(self.optimizer, lr_lambda=lr_lambda2)
        else:
            return LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)

    @property
    def _finest_mesh(self):
        return self._meshes[-1]