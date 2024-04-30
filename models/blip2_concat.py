from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import Blip2Processor, Blip2Model, AutoTokenizer
from util import base64str_to_PILobj
import torch
import copy
import torch.nn.functional as F
import torch

class CustomBLIP(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 tokenizer=None,
                 config=None,
                 freeze_image_encoder=True,
                 freeze_text_encoder=True,
                 map_dim=1024,
                 dropout_lst=[0.1, 0.4, 0.2]
                 ):
        super().__init__()

        self.map_dim = map_dim
        self.dropout_lst = dropout_lst
        self.num_mapping_layers = 5
        self.head = 'concat'

        self.tokenizer = tokenizer
        self.num_labels = 1

        self.blip = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")#, torch_dtype=torch.float16)
        # self.blip = torch.load("model_output/blip_entire_model_Salesforce-BlipModel-blip2-flan-t5-xlinn-concat.pt")

        # self.image_encoder = copy.deepcopy(self.blip.vision_model)
        # self.text_encoder = copy.deepcopy(self.blip.text_model)

        # Not using pretrained map
        image_map_layers = [nn.Linear(self.blip.vision_model.config.hidden_size, self.map_dim),
                            nn.Dropout(p=self.dropout_lst[0])]
        text_map_layers = [nn.Linear(self.blip.vision_model.config.projection_dim, self.map_dim),
                           nn.Dropout(p=self.dropout_lst[0])]
        # text_map_layers = [nn.Linear(self.blip.language_model.config.vocab_size, self.map_dim),
        #                    nn.Dropout(p=self.dropout_lst[0])] # For flank
        qformer_map_layers = [nn.Linear(self.blip.qformer.config.hidden_size, self.map_dim),
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
        self.qformer_map = nn.Sequential(*qformer_map_layers)

        if self.head == 'concat':
            pre_output_input_dim = self.map_dim*3
        elif self.head == 'cross':
            pre_output_input_dim = self.map_dim**3

        pre_output_layers = [nn.Dropout(p=self.dropout_lst[1])]
        pre_output_layers.extend([nn.Linear(pre_output_input_dim, self.map_dim),
                                  nn.ReLU(),
                                  nn.Dropout(p=self.dropout_lst[2])])

        self.pre_output = nn.Sequential(*pre_output_layers)
        self.classifier = nn.Linear(self.map_dim, 1)

        if freeze_image_encoder:
            for _, p in self.blip.vision_model.named_parameters():
                p.requires_grad_(False)

        if freeze_text_encoder:
            for _, p in self.blip.language_model.named_parameters():
                p.requires_grad_(False)
        
        for param in self.blip.parameters():
            param.requires_grad = False

        self.cross_entropy_loss = torch.nn.BCELoss(reduction='mean')
        # import pdb; pdb.set_trace()
        # self.classifier = nn.Linear(self.text_encoder.config.hidden_size + self.vision_encoder.config.hidden_size, 1) 
        # self.output_attentions = self.blip.config.output_attentions
        # self.output_hidden_states = (
        #     self.blip.config.output_hidden_states
        # )
        # del self.blip       

    def forward(self, batch, device):

        pixel_values = batch['pixel_values'].to(device)
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # output_attentions = self.blip.config.output_attentions
        # output_hidden_states = (
        #     self.blip.config.output_hidden_states
        # )

        image_embeds = self.blip.get_image_features(pixel_values=pixel_values.squeeze(1),
                                                    return_dict=True, output_hidden_states=True).pooler_output #[bs, 1408]
            # **batch)
        # import pdb; pdb.set_trace()
        text_embeds = self.blip.get_text_features(input_ids=input_ids.squeeze(1), 
                                                  attention_mask=attention_mask.squeeze(1),
                                                  return_dict=True, 
                                                  output_hidden_states=True
                                                  )[0][:,:,0]#[0] #[bs, 512, 50272]
        # text_embeds = self.blip.get_text_features(input_ids=input_ids.squeeze(1), 
        #                                           attention_mask=attention_mask.squeeze(1),
        #                                           return_dict=True, 
        #                                           output_hidden_states=True,
        #                                           decoder_input_ids=batch['pad_token_id'].reshape(batch['pad_token_id'].shape[0],1).to(device) #For Salesforce/blip2-flan-t5-xl only
        #                                           )[0].squeeze(1) # For flan
        
            # **batch)
        qformer_feas = self.blip.get_qformer_features(pixel_values=pixel_values.squeeze(1),
                                                    # input_ids=input_ids.squeeze(1), 
                                                    # attention_mask=attention_mask.squeeze(1),
                                                   return_dict=True, output_hidden_states=True).pooler_output #[bs, 768]
        image_features = self.image_map(image_embeds)
        # import pdb; pdb.set_trace()
        text_features = self.text_map(text_embeds)
        qformer_feas = self.qformer_map(qformer_feas)
        
        
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        qformer_feas = F.normalize(qformer_feas, p=2, dim=1)

        if self.head == 'concat':
            features = torch.cat([image_features, text_features, qformer_feas], dim=1)
        elif self.head == 'cross':
            ## TO CHECK
            features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1), qformer_feas.unsqueeze(0)) # [16, d, d]
            features = features.reshape(features.shape[0], -1)  # [16, d*d]

        features = self.pre_output(features)
        logits = self.classifier(features)
        prediction = torch.sigmoid(logits)

        return prediction

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

class BLIPProcessDataset(Dataset):
    def __init__(self, dataset):
        self.image_size = 224
        self.dataset = dataset
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-flan-t5-xl")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
       
        pixel_values = self.processor(images=base64str_to_PILobj(item["image"]).convert("RGB").resize((self.image_size, self.image_size)),
                                            return_tensors="pt")['pixel_values']
        
        text_output = self.tokenizer(text=item['text'],
                                     padding='max_length', 
                                     return_tensors="pt", 
                                     max_length=512,
                                     truncation=True
                                     )
        # print(text_output.keys())
        # import pdb; pdb.set_trace()

        label = torch.LongTensor([item['label']])

        return {
            'pixel_values': pixel_values,
            'input_ids': text_output['input_ids'],
            'attention_mask': text_output['attention_mask'],
            'labels': label,
            'idx_memes': item['id'],
            'image': item['image'],
            'pad_token_id': self.tokenizer.pad_token_id,
        }

# blip_entire_model_Salesforce-BlipModel-blip-image-captioning-base-inn.pt
# CUR > blip_entire_model_Salesforce-BlipModel-blip-image-captioning-large-inn.pt
