import torch.nn as nn
from einops import pack, unpack


from model.transformer import TransformerBlockAdaLNZero
from model.positional_emb import LearnedSinusoidalPosEmb

class ViTDecoder(nn.Module):

    def __init__(self, patch_dim, max_patches_len, fsq_n_levels, time_dim, d, nh, n_layers, n_registers, cond_d, num_classes):
        super().__init__()
        self.fsq_proj_up = nn.Linear(fsq_n_levels, d, bias=False)
        self.regr_pos_enc = nn.Embedding(n_registers, d)
        self.latent_pos_enc = nn.Embedding(max_patches_len, d)
        self.patch_proj = nn.Linear(patch_dim, d, bias=False)
        self.sinu_pos_emb = LearnedSinusoidalPosEmb(time_dim)
        fourier_dim = time_dim + 1
        self.time_mlp = nn.Sequential(
            self.sinu_pos_emb,
            nn.Linear(fourier_dim, cond_d),
            nn.SiLU(),
            nn.Linear(cond_d, cond_d)
        )
        # TODO: 1 step lookup vs proj up?
        # self.class_enc = nn.Sequential(
        #         nn.Embedding(num_classes, 32),
        #         nn.Linear(32, 64),
        #         nn.SiLU(),
        #         nn.Linear(64, cond_d)
        #     )

        self.blocks = nn.ModuleList(
            [TransformerBlockAdaLNZero(d, nh, cond_d) for _ in range(n_layers)]
        )
        self.proj_head = nn.Linear(d, patch_dim)
        

    def forward(self, registers, register_mask_token, noised_latent_patches, timesteps):
        b, regr_t, fsq_d = registers.shape
        registers = self.fsq_proj_up(registers)
        registers_pos_emb = self.regr_pos_enc.weight[:regr_t,:]
        registers = registers + registers_pos_emb

        latents = self.patch_proj(noised_latent_patches)
        latents_pos_emb = self.latent_pos_enc.weight
        latents = latents + latents_pos_emb

        x, ps = pack([registers, register_mask_token, latents], "b * d")
        
        t_emb = self.time_mlp(timesteps)
        # class_emb = self.class_enc(classes)
        conditioning = t_emb # + class_emb

        first_block = self.blocks[0]
        x = first_block(x, conditioning)

        _, _, first_layer_features = unpack(x, ps, "b * repa_dim")

        for block in self.blocks[1:]:
            x = block(x, conditioning)

        x = self.proj_head(x)
        _, _, denoised_latents = unpack(x, ps, "b * d_out")

        return denoised_latents, first_layer_features