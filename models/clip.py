from transformers import CLIPModel, AutoProcessor
from transformers import CLIPTokenizer, CLIPProcessor, AutoTokenizer
from util import base64str_to_PILobj
from torch.utils.data import Dataset
import torch.nn as nn
import os
import copy
import torch.nn.functional as F
import torch

class CLIPProcessDataset(Dataset):
    def __init__(self, dataset):

        self.dataset = dataset
        self.image_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.text_processor = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        pixel_values = self.image_processor(images=base64str_to_PILobj(item["image"]).convert("RGB"),
                                            return_tensors="pt")['pixel_values']

        text_output = self.text_processor(item['text'],
                                          padding='max_length', 
                                          return_tensors="pt", 
                                          truncation=True)
        label = torch.LongTensor([item['label']])

        return {
            'pixel_values': pixel_values,
            'input_ids': text_output['input_ids'],
            'attention_mask': text_output['attention_mask'],
            'labels': label,
            'idx_memes': item['id'],
            'image': item['image']
        }


class CLIPClassifier(nn.Module):

    def __init__(self, 
                 map_dim=32,
                 dropout_lst=[0.1, 0.4, 0.2],
                 pretrained_model='openai/clip-vit-large-patch14',
                 freeze_image_encoder=True,
                 freeze_text_encoder=True
                 ):
        super().__init__()

        self.map_dim = map_dim
        self.dropout_lst = dropout_lst
        self.num_mapping_layers = 5
        self.head = 'concat'

        self.clip = CLIPModel.from_pretrained(pretrained_model)
        self.image_encoder = copy.deepcopy(self.clip.vision_model)
        self.text_encoder = copy.deepcopy(self.clip.text_model)

        # Not using pretrained map
        image_map_layers = [nn.Linear(self.image_encoder.config.hidden_size, self.map_dim),
                            nn.Dropout(p=self.dropout_lst[0])]
        text_map_layers = [nn.Linear(self.text_encoder.config.hidden_size, self.map_dim),
                           nn.Dropout(p=self.dropout_lst[0])]
        for _ in range(1, self.num_mapping_layers):
            image_map_layers.extend([nn.ReLU(), 
                                     nn.Linear(self.map_dim, self.map_dim), 
                                     nn.Dropout(p=self.dropout_lst[0])])
            text_map_layers.extend([nn.ReLU(), 
                                    nn.Linear(self.map_dim, self.map_dim), 
                                    nn.Dropout(p=self.dropout_lst[0])])

        self.image_map = nn.Sequential(*image_map_layers)
        self.text_map = nn.Sequential(*text_map_layers)

        pre_output_input_dim = self.map_dim*2

        pre_output_layers = [nn.Dropout(p=self.dropout_lst[1])]
        pre_output_layers.extend([nn.Linear(pre_output_input_dim, self.map_dim),
                                  nn.ReLU(),
                                  nn.Dropout(p=self.dropout_lst[2])])

        self.pre_output = nn.Sequential(*pre_output_layers)
        self.output = nn.Linear(self.map_dim, 1)

        self.cross_entropy_loss = torch.nn.BCELoss(reduction='mean')

        if freeze_image_encoder:
            for _, p in self.image_encoder.named_parameters():
                p.requires_grad_(False)

        if freeze_text_encoder:
            for _, p in self.text_encoder.named_parameters():
                p.requires_grad_(False)
        del self.clip
        

    def forward(self, batch, device):
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # import pdb; pdb.set_trace()
        image_features = self.image_encoder(pixel_values=pixel_values.squeeze(1)).pooler_output
        image_features = self.image_map(image_features)
        # import pdb; pdb.set_trace()
        text_features = self.text_encoder(input_ids=input_ids.squeeze(1),
                                          attention_mask=attention_mask.squeeze(1)).pooler_output
        
        text_features = self.text_map(text_features)

        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        if self.head == 'concat':
            features = torch.cat([image_features, text_features], dim=1)
        elif self.head == 'cross':
            features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1)) # [16, d, d]
            features = features.reshape(features.shape[0], -1)  # [16, d*d]

        features = self.pre_output(features)
        logits = self.output(features)
        preds = torch.sigmoid(logits)

        return preds