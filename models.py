import torch
import torch.nn as nn
import torch.nn.functional as F
        
class Linker(nn.Module):
    def __init__(
        self,
        n_features=4,
        link_width=256,
        link_n_head=4,
        link_n_layers=12,
        link_fc=1024,
        pdg_emb=8,
        device="cuda:0",
        num_pdg=540
    ):
        super().__init__()
        self.pdg_emb = pdg_emb
        self.num_pdg = num_pdg
        self.pdg_embedder = nn.Embedding(num_pdg + 1,pdg_emb)
        self.projector = nn.Linear(n_features + pdg_emb, link_width)
        self.device = device
        # Encoders
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=link_width, nhead=link_n_head, dim_feedforward=link_fc, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=link_n_layers)

        self.to(device)
        
    def forward(self, dataset):
        # prepare
        pdg_x = self.pdg_embedder(dataset['pdg_x'].to(self.device))
        feat_x = dataset['feature_x'].to(self.device)
        pdg_y = self.pdg_embedder(dataset['pdg_y'].to(self.device))
        feat_y = dataset['feature_y'].to(self.device)
        padding_mask = dataset['padding_mask'].to(self.device)
        
        encoder_input_x = self.projector(torch.cat([pdg_x, feat_x], axis=-1))
        encoder_input_y = self.projector(torch.cat([pdg_y, feat_y], axis=-1))
        # Encoders
        encoded_x = self.encoder(encoder_input_x, src_key_padding_mask=~padding_mask)
        encoded_y = self.encoder(encoder_input_y, src_key_padding_mask=~padding_mask)
        return self.corr_matrix(encoded_x, encoded_y, padding_mask[...,None])
        # output: [batch, max_seq_len, max_seq_len]

    def corr_matrix(self, data_x, data_y, mask):
        M_ix = (data_x*mask).unsqueeze(2).repeat(1,1,data_x.shape[1],1)
        M_iy = (data_y*mask).unsqueeze(1).repeat(1,data_y.shape[1],1,1)
#         return torch.einsum('ixya,ixyb->ixy',M_ix,M_iy) # M_ixy
        return F.cosine_similarity(M_ix, M_iy, dim=-1, eps=1e-6)

class HyperEmbedder(torch.nn.Module):
    def __init__(
        self,
        n_features=11,
        tr_width=64,
        tr_n_head=8,
        tr_n=4,
        tr_hidden_size=2048,
        pdg_emb=5,
        dim_hyper=3,
        num_pdg=40,
        device="cuda:0"
    ):
        super().__init__()
        self.device = device
        self.pdg_emb = pdg_emb
        self.tr_width = tr_width
        self.pdg_embedder = nn.Embedding(num_pdg + 1,pdg_emb)
        self.projector = nn.Linear(in_features=n_features + pdg_emb, out_features=tr_width)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=tr_width, nhead=tr_n_head, dim_feedforward=tr_hidden_size, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=tr_n)
        self.particle_importance = nn.Linear(in_features=tr_width, out_features=1)
        self.phi_vector = torch.nn.Linear(in_features=tr_width, out_features=dim_hyper)

        self.phi_norm = torch.nn.Linear(in_features=tr_width, out_features=1)
        torch.nn.init.xavier_uniform_(self.phi_norm.weight, gain=torch.nn.init.calculate_gain('sigmoid'))
        self.to(device)
        
    def forward(self, dataset):
        pdg = self.pdg_embedder(dataset['pdg'].to(self.device))
        feat = dataset['feature'].to(self.device)
        padding_mask = dataset['padding_mask'].to(self.device)
        att_input = self.projector(torch.cat([pdg, feat], axis=-1))
        # src_key_padding_mask is inversely defined!!! True = Skip, False = Keep
        transformed = self.transformer_encoder(att_input, src_key_padding_mask=~padding_mask) # features: [batch, n_particles, width]
        particle_weights = F.softmax(
            self.particle_importance(transformed).squeeze(-1).masked_fill_(~padding_mask,-1e9),
            dim=-1)
        features = F.adaptive_avg_pool1d(
            transformed.permute(0,2,1) * particle_weights.unsqueeze(1), 1
            ).squeeze(-1) # euclidean output: [batch, width]
        v = F.normalize(self.phi_vector(features))
        p = torch.sigmoid(self.phi_norm(features))
        return p * v # hyperbolic output: [batch, dim_hyper]        
    
class Reconstructor(nn.Module):
    def __init__(
        self,
        n_features=4,
        gen_tr_width=64,
        gen_encoder_n_head=8,
        gen_encoder_n_layers=4,
        gen_encoder_fc=2048,
        gen_decoder_n_head=8,
        gen_decoder_n_layers=4,
        gen_decoder_fc=2048,
        pdg_emb=5,
        dim_hyper=3,
        device="cuda:0",
        num_pdg=540
    ):
        super().__init__()
        self.pdg_emb = pdg_emb
        self.gen_tr_width = gen_tr_width
        self.num_pdg = num_pdg
        self.pdg_embedder = nn.Embedding(num_pdg + 1,pdg_emb)
        self.device = device
        self.decoder_broadener = 2
        # Encoder
        self.projector = nn.Linear(in_features=n_features+pdg_emb, out_features=gen_tr_width)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=gen_tr_width, nhead=gen_encoder_n_head, dim_feedforward=gen_encoder_fc, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=gen_encoder_n_layers)
        self.particle_importance = nn.Linear(in_features=gen_tr_width, out_features=1)
        # self.phi_vector = torch.nn.Linear(in_features=gen_tr_width, out_features=dim_hyper)
        # self.phi_norm = torch.nn.Linear(in_features=gen_tr_width, out_features=1)

        decoder_width = self.decoder_broadener * gen_tr_width
        # Decoder
        self.projector_decoder = nn.Linear(in_features=n_features+pdg_emb+gen_tr_width, out_features=decoder_width)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_width, nhead=gen_decoder_n_head, dim_feedforward=gen_decoder_fc, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=gen_decoder_n_layers)
        
        # Multi-task heads
        # PDG Head 
        self.pdg_head = nn.Sequential(
            nn.Linear(decoder_width, 256), nn.ReLU(), 
            nn.Linear(256, 64), nn.ReLU(), 
            nn.Linear(64,num_pdg+1)
        )         
        self.combined_head = nn.Linear(decoder_width+num_pdg+1, 256)
        self.p_head = nn.Sequential(
            nn.Linear(256,64), nn.ReLU(),
            nn.Linear(64,8), nn.ReLU(),
            nn.Linear(8,3)
        )         
        self.e_head = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(256,64), nn.ReLU(),
            nn.Linear(64,8), nn.ReLU(),
            nn.Linear(8,1), nn.ReLU()
        )
        
        self.to(device)
        
    def forward(self, dataset):
        # prepare
        pdg = self.pdg_embedder(dataset['pdg_x'].to(self.device))
        feat = dataset['feature_x'].to(self.device)
        # emb = dataset['emb'].to(self.device)
        padding_mask = dataset['padding_mask'].to(self.device)
        particle_info = torch.cat([pdg, feat], axis=-1)
        att_input = self.projector(particle_info)
        # main 
        # src_key_padding_mask is inversely defined!!! True = Skip, False = Keep
        encoded_particle = self.transformer_encoder(att_input, src_key_padding_mask=~padding_mask)
        
        # embedding
        particle_weights = F.softmax(
            self.particle_importance(encoded_particle).squeeze(-1).masked_fill_(~padding_mask,-1e9),
            dim=-1)
        features = F.adaptive_avg_pool1d(
            encoded_particle.permute(0,2,1) * particle_weights.unsqueeze(1), 1
            ).squeeze(-1) # euclidean output: [batch, width]
        # v = F.normalize(self.phi_vector(features))
        # p = torch.sigmoid(self.phi_norm(features))
        # emb = p * v

        # Decoder
        att_input = torch.cat([particle_info,features.unsqueeze(1).repeat((1,padding_mask.shape[1],1))],axis=-1)
        att_input = self.projector_decoder(att_input)
        decoded_particle = self.transformer_decoder(
            att_input, encoded_particle.repeat(1,1,self.decoder_broadener), 
            tgt_key_padding_mask=~padding_mask, memory_key_padding_mask=~padding_mask
        )
        
        # PDG head
        pdg_out = self.pdg_head(decoded_particle)
        # feature head
        combined = self.combined_head(torch.cat([decoded_particle,pdg_out.detach()],axis=-1))
        feat_out = torch.cat(
            [self.p_head(combined),self.e_head(combined)], axis=-1
            )
        
        return pdg_out, feat_out
        # output: [batch, max_seq_len, num_pdg+1], [batch, max_seq_len, 4]