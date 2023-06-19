from typing import Optional
import torch
from torch import nn
from pytorch_pretrained_vit import ViT


class vitGeno(ViT):

    def __init__(
            self,
            name: Optional[str] = None,
            pretrained: bool = False,
            patches: int = 16,
            dim: int = 768,
            ff_dim: int = 3072,
            num_heads: int = 12,
            num_layers: int = 12,
            attention_dropout_rate: float = 0.0,
            dropout_rate: float = 0.1,
            representation_size: Optional[int] = None,
            load_repr_layer: bool = False,
            classifier: str = 'token',
            positional_embedding: str = '1d',
            in_channels: int = 3,
            image_size: Optional[int] = None,
            num_classes: Optional[int] = None,
    ):

        # init parent
        # super().__init__()
        super().__init__(name, pretrained, patches, dim, ff_dim, num_heads, num_layers,
                         attention_dropout_rate, dropout_rate, representation_size,
                         load_repr_layer, classifier, positional_embedding, in_channels,
                         image_size, num_classes)

        # add multiple fc
        # number of classes can be changed later
        del self.fc
        self.fc1 = nn.Linear(dim, 2)
        self.fc2 = nn.Linear(dim, 2)

    # redefine forward
    def forward(self, x):
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)  # b,gh*gw+1,d
        x = self.transformer(x)  # b,gh*gw+1,d
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(self, 'fc1'):
            x = self.norm(x)[:, 0]  # b,d
            x1 = self.fc1(x)  # b,num_classes
            x2 = self.fc2(x)  # b,num_classes
        return x1, x2
