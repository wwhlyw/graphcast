from .layer import Encoder, Processor, Decoder
import torch.nn as nn


class GraphCastNet(nn.Module):
    def __init__(
        self,
        vg_in_channels,
        vg_out_channels,
        vm_in_channels,
        em_in_channels,
        eg2m_in_channels,
        em2g_in_channels,
        latent_dims,
        processing_steps,
        g2m_src_idx,
        g2m_dst_idx,
        m2m_src_idx,
        m2m_dst_idx,
        m2g_src_idx,
        m2g_dst_idx,
        mesh_node_feats,
        mesh_edge_feats,
        g2m_edge_feats,
        m2g_edge_feats,
        clip_edge_idx,
        mesh_connects,
        per_variable_level_mean,
        per_variable_level_std,
    ):
        super().__init__()
        self.vg_out_channels = vg_out_channels
        self.mesh_node_feats = mesh_node_feats
        self.mesh_edge_feats = mesh_edge_feats
        self.g2m_edge_feats = g2m_edge_feats
        self.m2g_edge_feats = m2g_edge_feats
        self.per_variable_level_mean = per_variable_level_mean
        self.per_variable_level_std = per_variable_level_std
        self.encoder = Encoder(
            vg_in_channels=vg_in_channels,
            vm_in_channels=vm_in_channels,
            em_in_channels=em_in_channels,
            eg2m_in_channels=eg2m_in_channels,
            em2g_in_channels=em2g_in_channels,
            latent_dims=latent_dims,
            src_idx=g2m_src_idx,
            dst_idx=g2m_dst_idx
        )
        self.processor = Processor(
            node_in_channels=latent_dims,
            node_out_channels=latent_dims,
            edge_in_channels=latent_dims,
            edge_out_channels=latent_dims,
            processing_steps=processing_steps,
            latent_dims=latent_dims,
            src_idx=m2m_src_idx,
            dst_idx=m2m_dst_idx,
            clip_edge_idx=clip_edge_idx,
            mesh_connects=mesh_connects
        )
        self.decoder = Decoder(
            node_in_channels=latent_dims,
            node_out_channels=latent_dims,
            edge_in_channels=latent_dims,
            edge_out_channels=latent_dims,
            node_final_dims=vg_out_channels,
            latent_dims=latent_dims,
            src_idx=m2g_src_idx,
            dst_idx=m2g_dst_idx
        )
    def forward(self, grid_node_feats):
        # print(self.mesh_node_feats.shape)
        # print(self.mesh_edge_feats.shape)
        # print(self.g2m_edge_feats.shape)
        # print(self.m2g_edge_feats.shape)
    
        vg, vm, em, _, em2g = self.encoder(
            grid_node_feats,
            self.mesh_node_feats,
            self.mesh_edge_feats,
            self.g2m_edge_feats,
            self.m2g_edge_feats    
        )
        updated_vm, _ = self.processor(vm, em)
        node_feats = self.decoder(vg, updated_vm, em2g)


        output = (node_feats * self.per_variable_level_std + self.per_variable_level_mean) + \
            grid_node_feats[:, :, self.vg_out_channels: 2 * self.vg_out_channels]
        return output