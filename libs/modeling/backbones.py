import torch
from torch import nn
from torch.nn import functional as F

from .models import register_backbone
from .blocks import (get_sinusoid_encoding, TransformerBlock, MaskedConv1D,
                     ConvBlock, LayerNorm, AggregationBlock, TemporalMaxer)


@register_backbone("convTransformer")
class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        n_in,                  # input feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        arch = (2, 2, 5),      # (#convs, #stem transformers, #branch transformers)
        mha_win_size = [-1]*6, # size of local window for mha
        scale_factor = 2,      # dowsampling rate for the branch
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
        use_rel_pe = False,    # use relative position embedding
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(mha_win_size) == (1 + arch[2])
        self.n_in = n_in
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe

        # feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # self.tmx = TemporalMaxer(3, 2, 1, n_embd)

        self.embd = MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
        self.embd_norm = LayerNorm(n_embd)

        self.learn_embd = MaskedConv1D(
            n_in, n_embd, n_embd_ks,
            stride=1, padding=n_embd_ks // 2, bias=(not with_ln)
        )
        self.learn_embd_norm = LayerNorm(n_embd)

        self.learn_embd2 = MaskedConv1D(
            n_in, n_embd, n_embd_ks,
            stride=1, padding=n_embd_ks // 2, bias=(not with_ln)
        )
        self.learn_embd_norm2 = LayerNorm(n_embd)

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        self.long_attn = nn.ModuleList()

        for idx in range(arch[2]):
            self.stem.append(
                AggregationBlock(
                    n_embd, n_head
                )
            )

        for idx in range(arch[1]):
            self.long_attn.append(
                AggregationBlock(
                    n_embd, n_head
                )
            )

        self.tmx = TemporalMaxer(
                    n_embd_ks,
                    stride=4,
                    padding=n_embd_ks // 2,
                    n_embd=n_embd
                )


        self.convblock = nn.ModuleList()
        # self.convblock2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(arch[2]):
            self.convblock.append(
                ConvBlock(n_embd)
            )
            self.norm.append(
                LayerNorm(n_embd)
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        self.aggr = nn.ModuleList()
        self.down_sample = nn.ModuleList()

        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe
                )
            )

            self.down_sample.append(
                MaskedConv1D(
                    n_embd, n_embd, n_embd_ks,
                    stride=2, padding=n_embd_ks // 2, bias=(not with_ln)
                )
            )

            self.aggr.append(
                AggregationBlock(
                    n_embd, n_head
                )
            )

        self.long_mem = nn.Embedding(n_embd, 192)
        self.long_query = nn.Embedding(n_embd, 192 // 2)

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )

        f, f_mask = self.learn_embd(x, mask)
        f = self.relu(self.learn_embd_norm(f))
        # f, f_mask = self.tmx(f, f_mask)

        b, d, n = f.shape
        long_mask = torch.ones((b, 1, 192), device=f.device)
        long_mask2 = torch.ones((b, 1, 192 // 2), device=f.device)
        emp_q = self.long_mem.weight.repeat(f.shape[0], 1, 1)
        emp_q2 = self.long_query.weight.repeat(f.shape[0], 1, 1)

        emp_q, long_mask = self.long_attn[0](emp_q, long_mask, f, f_mask)
        emp_q2, long_mask2 = self.long_attn[1](emp_q2, long_mask2, emp_q, long_mask)

        f2, f2_mask = self.learn_embd2(x, mask)
        f2 = self.relu(self.learn_embd_norm2(f2))
        # f2, f2_mask = self.tmx(x, mask)
        # # f2 = self.relu(self.learn_embd_norm2(f2))

        x, mask = self.embd(x, mask)
        x = self.relu(self.embd_norm(x))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # prep for outputs
        out_feats = (x, )
        out_masks = (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            b, n, t = x.shape
            # len_split = t // 2
            # x1_mask, x2_mask = mask[:, :, :len_split], mask[:, :, len_split:]
            # x1, x2 = x[:, :, :len_split], x[:, :, len_split:]
            # x1, x1_mask = self.stem[idx](x1, x1_mask, f2, f2_mask)
            # x2, x2_mask = self.stem[idx](x2, x2_mask, f2, f2_mask)
            # x, mask = torch.cat([x1, x2], dim=2), torch.cat([x1_mask, x2_mask], dim=2)

            len_term = t // 4
            x1_mask, x2_mask, x3_mask, x4_mask = mask[:, :, :len_term], mask[:, :, len_term:len_term*2], mask[:, :, len_term*2:len_term*3], mask[:, :, len_term*3:]
            x1, x2, x3, x4 = x[:, :, :len_term], x[:, :, len_term:len_term*2], x[:, :, len_term*2:len_term*3], x[:, :, len_term*3:]

            x1, x1_mask = self.stem[idx](x1, x1_mask, f2, f2_mask)
            x2, x2_mask = self.stem[idx](x2, x2_mask, f2, f2_mask)
            x3, x3_mask = self.stem[idx](x3, x3_mask, f2, f2_mask)
            x4, x4_mask = self.stem[idx](x4, x4_mask, f2, f2_mask)

            x, mask = torch.cat([x1, x2, x3, x4], dim=2), torch.cat([x1_mask, x2_mask, x3_mask, x4_mask], dim=2)
            x, mask = self.convblock[idx](x, mask)

            x, mask = self.aggr[idx](x, mask, emp_q2, long_mask2)

            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks


@register_backbone("conv")
class ConvBackbone(nn.Module):
    """
        A backbone that with only conv
    """
    def __init__(
        self,
        n_in,               # input feature dimension
        n_embd,             # embedding dimension (after convolution)
        n_embd_ks,          # conv kernel size of the embedding network
        arch = (2, 2, 5),   # (#convs, #stem convs, #branch convs)
        scale_factor = 2,   # dowsampling rate for the branch
        with_ln=False,      # if to use layernorm
    ):
        super().__init__()
        assert len(arch) == 3
        self.n_in = n_in
        self.arch = arch
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

        # feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # stem network using convs
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(ConvBlock(n_embd, 3, 1))

        # main branch using convs with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(ConvBlock(n_embd, 3, self.scale_factor))

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # stem conv
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (x, )
        out_masks = (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks