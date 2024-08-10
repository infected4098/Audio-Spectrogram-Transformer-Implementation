from modules.transformer import TransformerBlock

class AST(nn.Module):
    """
    label_dim: the number of total classes, 41
    fstride: the stride of patch spliting on the frequency dimension
    tstride: the stride of patch spliting on the time dimension
    input_fdim: # frequency bins of the input spectrogram
    input_tdim: # time frames of the input spectrogram
    embed_dim : # embed_dimensions of ViT
    """
    def __init__(self, label_dim=41, fstride=10, tstride=10, input_fdim=128, input_tdim=100, embed_dim = 384, num_heads = 6, n_blocks = 12, verbose=True):

        super(AST, self).__init__()

        self.label_dim = label_dim
        self.fstride = fstride
        self.tstride = tstride
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_blocks = n_blocks


        if verbose == True:
            print('---------------AST Model Initializing---------------')



        self.patch_embedding = nn.Conv2d(in_channels = 1, out_channels = self.embed_dim, kernel_size = 16, stride = (self.fstride, self.tstride))

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.patch_embedding.apply(self.init_weights)
        f_dim, t_dim = self.get_shape(fstride = self.fstride, tstride = self.tstride,
                       input_fdim = self.input_fdim, input_tdim = self.input_tdim)
        num_patches = f_dim * t_dim
        self.position_embeddings = nn.Parameter(torch.randn(num_patches + 1, self.embed_dim))

        self.blocks = nn.ModuleList() # Transformer block 만들기.
        for _ in range(self.n_blocks):
          self.blocks.append(TransformerBlock(self.embed_dim, self.num_heads))


        self.norm = nn.LayerNorm(self.embed_dim)
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.embed_dim), nn.Linear(self.embed_dim, self.label_dim))


        if verbose == True:

            print('frequency stride={:d}, time stride={:d}'.format(self.fstride, self.tstride))
            print('number of patches={:d}'.format(num_patches))



    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=100):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.embed_dim, kernel_size=16, stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def init_weights(self, m, mean = 0.0, std = 0.01):
      classname = m.__class__.__name__
      if classname.find("Conv") != -1: # if module == Conv
        m.weight.data.normal_(mean,std) #Initialize weights

    @autocast() #Speed up Training
    def forward(self, x):
        """
        x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        return: prediction
        """
        # expect input x = (batch_size, frequency bins, time bins), e.g., (16, 128, 100)
        x = x.unsqueeze(1) # (batch_size, 1, frequency bins, time bins), e.g., (16, 1, 128, 100)

        B = x.shape[0]
        x = self.patch_embedding(x) # (batch_size, #embed_dim, f_dim, t_dim)
        x = x.flatten(2) # (batch_size, #embed_dim, num_patches)
        x = x.transpose(1, 2) #(batch_size, num_patches, #embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, #embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, #embed_dim)
        x = x + self.position_embeddings #(B, num_patches + 1, #embed_dim)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2 # (B, 1, #embed dim).

        x = self.mlp_head(x) # Why no Softmax in the end? https://slamwithme.oopy.io/305bb7e0-1062-4785-a82e-9e2a5debd0f4 
        return x
