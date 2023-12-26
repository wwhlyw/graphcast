import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter


class MLPNet(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dims, has_layernorm=True):
        super().__init__()
        cell_list = [
            nn.Linear(in_channels, latent_dims, bias=False),
            nn.SiLU(),
            nn.Linear(latent_dims, out_channels, bias=False),
        ]
        if has_layernorm:
            cell_list.append(nn.LayerNorm([out_channels]))
        self.network = nn.Sequential(*cell_list)

    def forward(self, x):
        return self.network(x)     


class InteractionLayer(nn.Module):
    def __init__(
            self,
            node_in_channels,
            node_out_channels,
            edge_in_channels,
            edge_out_channels,
            latent_dims,
            src_idx,
            dst_idx,
            is_homo,
        ):
        super().__init__()

        # process node
        self.node_fn = MLPNet(in_channels=node_in_channels + edge_out_channels, out_channels=node_out_channels, latent_dims=latent_dims)

        # process edge
        self.edge_fn = MLPNet(in_channels=2 * node_in_channels + edge_in_channels, out_channels=edge_out_channels, latent_dims=latent_dims)

        self.src_idx = src_idx
        self.dst_idx = dst_idx
        self.is_homo = is_homo
    
    def forward(self, feats):
        if self.is_homo:
            src_node_feats, dst_node_feats, edge_feats = feats[0], feats[0], feats[1]
        else:
            src_node_feats, dst_node_feats, edge_feats = feats[0], feats[1], feats[2]

        # [batch, grid_num, feat]
        src_feats = torch.index_select(src_node_feats, dim=1, index=self.src_idx)

        dst_feats = torch.index_select(dst_node_feats, dim=1, index=self.dst_idx)
        update_edge_feats = self.edge_fn(torch.concat((src_feats, dst_feats, edge_feats), axis=-1))
        print('self.dst_idx:',self.dst_idx.shape)
        sum_edge_feats = scatter(edge_feats, self.dst_idx, dim=1, reduce='sum')
        print('sum_edge_feats:',sum_edge_feats.shape)
        print('dst_node_feats', dst_node_feats.shape)
        update_dst_feats = self.node_fn(torch.concat((dst_node_feats, sum_edge_feats), axis=-1))
 
        return (update_dst_feats + dst_node_feats, update_edge_feats + edge_feats)
     

class Embedder(nn.Module):
    def __init__(
            self, 
            vg_in_channels, 
            vm_in_channels, 
            em_in_channels, 
            eg2m_in_channels, 
            em2g_in_channels, 
            latent_dims,
        ):
        super().__init__()
        self.vg_embedder = MLPNet(in_channels=vg_in_channels, out_channels=latent_dims, latent_dims=latent_dims)
        self.vm_embedder = MLPNet(in_channels=vm_in_channels, out_channels=latent_dims, latent_dims=latent_dims)
        self.em_embedder = MLPNet(in_channels=em_in_channels, out_channels=latent_dims, latent_dims=latent_dims)
        self.eg2m_embedder = MLPNet(in_channels=eg2m_in_channels, out_channels=latent_dims, latent_dims=latent_dims)
        self.em2g_embedder = MLPNet(in_channels=em2g_in_channels, out_channels=latent_dims, latent_dims=latent_dims)

    def forward(self, grid_node_feats, mesh_node_feats, mesh_edge_feats, g2m_edge_feats, m2g_edge_feats):
        v_g = self.vg_embedder(grid_node_feats)
        v_m = self.vm_embedder(mesh_node_feats)
        e_m = self.em_embedder(mesh_edge_feats)
        e_g2m = self.eg2m_embedder(g2m_edge_feats)
        e_m2g = self.em2g_embedder(m2g_edge_feats)

        return v_g, v_m, e_m, e_g2m, e_m2g


class G2MGnn(nn.Module):
    def __init__(
            self,
            node_in_channels,
            node_out_channels,
            edge_in_channels,
            edge_out_channels,
            latent_dims,
            src_idx,
            dst_idx,
        ):
        super().__init__()
        self.interaction = InteractionLayer(
            node_in_channels,
            node_out_channels,
            edge_in_channels,
            edge_out_channels,
            latent_dims,
            src_idx,
            dst_idx,
            is_homo=False,
        )
        self.grid_node_mlp = MLPNet(
            in_channels = node_in_channels,
            out_channels = node_out_channels,
            latent_dims = latent_dims,
        )

    def forward(self, grid_node_feats, mesh_node_feats, g2m_edge_feats):
        mesh_node_aggr, g2m_edge_attr = self.interaction((grid_node_feats, mesh_node_feats, g2m_edge_feats))
        grid_node_new = self.grid_node_mlp(grid_node_feats)
        return grid_node_new + grid_node_feats, mesh_node_aggr, g2m_edge_attr
    
class Encoder(nn.Module):
    def __init__(
        self,
        vg_in_channels,
        vm_in_channels,
        em_in_channels,
        eg2m_in_channels,
        em2g_in_channels,
        latent_dims,
        src_idx,
        dst_idx,
    ):
        super().__init__()
        self.feature_embedder = Embedder(
            vg_in_channels,
            vm_in_channels,
            em_in_channels,
            eg2m_in_channels,
            em2g_in_channels,
            latent_dims,
        )
        self.g2m_gnn = G2MGnn(
            node_in_channels=latent_dims,
            node_out_channels=latent_dims,
            edge_in_channels=latent_dims,
            edge_out_channels=latent_dims,
            latent_dims=latent_dims,
            src_idx=src_idx,
            dst_idx=dst_idx,
        )

    def forward(
        self, 
        grid_node_feats, 
        mesh_node_feats, 
        mesh_edge_feats, 
        g2m_edge_feats, 
        m2g_edge_feats
    ):

        vg, vm, em, eg2m, em2g = self.feature_embedder(grid_node_feats,
                                                       mesh_node_feats,
                                                       mesh_edge_feats,
                                                       g2m_edge_feats,
                                                       m2g_edge_feats) 
        
        vg, vm, eg2m = self.g2m_gnn(vg, vm, eg2m) 
        return vg, vm, em, eg2m, em2g      


class Processor(nn.Module):
    def __init__(
        self,
        node_in_channels,
        node_out_channels,
        edge_in_channels,
        edge_out_channels,
        processing_steps,
        latent_dims,
        src_idx,
        dst_idx,
    ):
        super().__init__()
        self.processing_steps = processing_steps
        self.cell_list = nn.Sequential()
        for _ in range(self.processing_steps):
            self.cell_list.append(InteractionLayer(
                node_in_channels,
                node_out_channels,
                edge_in_channels,
                edge_out_channels,
                latent_dims,
                src_idx,
                dst_idx,
                is_homo=True
            ))

    def forward(self, node_feats, edge_feats):
        node_feats, edge_feats = self.cell_list((node_feats, edge_feats))
        return node_feats, edge_feats
    

class M2Gnn(nn.Module):
    def __init__(
        self,
        node_in_channels,
        node_out_channels,
        edge_in_channels,
        edge_out_channels,
        latent_dims,
        src_idx, 
        dst_idx
    ):
        super().__init__()
        self.interaction = InteractionLayer(node_in_channels,
                                            node_out_channels,
                                            edge_in_channels,
                                            edge_out_channels,
                                            latent_dims,
                                            src_idx,
                                            dst_idx,
                                            is_homo=False)
        self.mesh_node_mlp = MLPNet(in_channels=node_in_channels,
                                    out_channels=node_out_channels,
                                    latent_dims=latent_dims)
        
    def forward(self, m2g_edge_feats, mesh_node_feats, grid_node_feats):
        grid_node_aggr, m2g_edge_aggr = self.interaction((mesh_node_feats, grid_node_feats, m2g_edge_feats))
        mesh_node_aggr = self.mesh_node_mlp(mesh_node_feats)
        return grid_node_aggr, mesh_node_aggr + mesh_node_feats, m2g_edge_aggr
    
class Decoder(nn.Module):
    def __init__(
        self,
        node_in_channels,
        node_out_channels,
        edge_in_channels,
        edge_out_channels,
        node_final_dims,
        latent_dims,
        src_idx,
        dst_idx,
    ):
        super().__init__()
        self.m2g_gnn = M2Gnn(node_in_channels,
                             node_out_channels,
                             edge_in_channels,
                             edge_out_channels,
                             latent_dims,
                             src_idx,
                             dst_idx,)
        self.node_fn = MLPNet(in_channels=node_out_channels,
                              out_channels=node_final_dims,
                              latent_dims=latent_dims,
                              has_layernorm=False)
        
    def forward(self, grid_node_feats, mesh_node_feats, m2g_edge_feats):
        grid_node_feats, mesh_node_feats, m2g_edge_feats = self.m2g_gnn(m2g_edge_feats,
                                                                        mesh_node_feats,
                                                                        grid_node_feats,
                                                                        )
        return self.node_fn(grid_node_feats)
    

