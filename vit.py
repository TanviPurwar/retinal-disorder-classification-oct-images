import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

image_size = 256
patch_size = int(256/16)
num_patches = int((image_size*image_size)/(patch_size*patch_size))
num_channels = 1
patch_dim = num_channels * patch_size * patch_size
num_classes = 4

# Finetune parameters
dim = 64
heads = 8
mlp_dim=128
depth = 8
dim_head = 64
dropout = 0.
emb_dropout = 0.

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        
        return self.net(x)

class Attention(nn.Module):
    
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim) 

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
    def forward(self, x):
 
        #create tuple with 3 elements or chunks
        qkv = self.to_qkv(x).chunk(3, dim = -1) #qkv.shape -> torch.Size([1, 257, 1536]) after chunk qkv[0].shape -> torch.Size([1, 257, 512])
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) #q.shape, k.shape, v.shape -> torch.Size([1, 8, 257, 64]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale #dots.shape -> torch.Size([1, 8, 257, 257]); (k.transpose).shape -> torch.Size([1, 8, 64, 257])

        attn = self.attend(dots) #softmax applied for attention
        attn = self.dropout(attn) #dropout applied 

        out = torch.matmul(attn, v) #out.shape -> torch.Size([1, 8, 257, 64])
        out = rearrange(out, 'b h n d -> b n (h d)') #out.shape -> torch.Size([1, 257, 512])
        
        return self.to_out(out) #return shape -> torch.Size([1, 257, 64])

class PreNorm(nn.Module):
    
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) #return normalized vector from attention function

class Transformer(nn.Module):
    
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
            
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            
        return x

class ViT(nn.Module):
    
    def __init__(self, *, patch_size, patch_dim, num_patches, dim, emb_dropout, heads, num_classes):
        super().__init__()
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size), #rearranges in patches output -> (b, num_patches, patch_dim)
            nn.LayerNorm(patch_dim), 
            nn.Linear(patch_dim, dim), #convert from (1, num_patches, patch_dim) -> (1, num_patches, dim)
            nn.LayerNorm(dim),
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        self.dropout = nn.Dropout(emb_dropout)
        
        # transformer model
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        self.to_latent = nn.Identity()
        
        # multi layer perceptron
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, img):
        x = self.to_patch_embedding(img) #x.shape -> (1, num_patches, dim)
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b) #remove it???
        
        #prepend learnable class embedding
        x = torch.cat((cls_tokens, x), dim=1) # x.shape -> (1, num_patches+1, dim)
        
        x += self.pos_embedding[:, :(n + 1)] # add values of pos_embedding to x
        
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = x[:, 0] #x.shape -> torch.Size([1, 64])
        
        x = self.to_latent(x)
    
        return self.mlp_head(x) #return shape -> torch.Size([1, 4])

'''
img = torch.randn(64, 1, 256, 256)

print(img.shape)

model = ViT(patch_size=patch_size, patch_dim=patch_dim, num_patches=num_patches, dim=dim, emb_dropout=emb_dropout, heads=heads, num_classes=num_classes)

print(model)

print('shape of output tensor: ', model(img).shape)   
'''     