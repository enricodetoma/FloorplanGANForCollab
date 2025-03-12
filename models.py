import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class LayoutTransformer(nn.Module):
    def __init__(self, dataset):
        super(LayoutTransformer, self).__init__()
        self.class_num = dataset.enc_len  # Number of room types
        self.element_num = dataset.maximum_elements_num  # Max elements per floor plan

        d_model = 256  # Embedding size for Transformer
        nhead = 8  # Number of attention heads
        num_layers = 6  # Number of Transformer layers

        # Transformer Encoder replaces CNN layers
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        # Decoder layers for room layout generation
        self.decoder_fc0 = nn.Linear(d_model, 128)
        self.decoder_fc1 = nn.Linear(128, 64)
        self.decoder_fc2 = nn.Linear(64, 4)  # Output: (x, y, width, height)

        # Room classification layer
        self.room_classifier = nn.Linear(64, self.class_num)

    def forward(self, x):
        # Transformer expects (Sequence, Batch, Features) format
        x = x.permute(1, 0, 2)  
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  

        # Decode to layout and classify rooms
        x = F.relu(self.decoder_fc0(x))
        x = F.relu(self.decoder_fc1(x))
        layout_output = self.decoder_fc2(x)  # (x, y, width, height)
        room_classes = self.room_classifier(x)  # Room type classification

        return layout_output, room_classes


class WireframeDiscriminator(nn.Module):
    def __init__(self, dataset):
        super(WireframeDiscriminator, self).__init__()
        self.class_num = dataset.enc_len
        self.element_num = dataset.maximum_elements_num

        d_model = 256  # Embedding size
        nhead = 8  # Number of attention heads
        num_layers = 3  # Fewer layers for Discriminator

        # Transformer-based Discriminator
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        self.transformer_discriminator = TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, 1)  # Binary classification (Real/Fake)

    def forward(self, x):
        # Process floor plan features with Transformer
        x = x.permute(1, 0, 2)
        x = self.transformer_discriminator(x)
        x = x.permute(1, 0, 2)

        # Final classification
        x = torch.mean(x, dim=1)  # Global average pooling
        x = self.fc(x)
        return x
