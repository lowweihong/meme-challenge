from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import BlipForConditionalGeneration, BlipForQuestionAnswering, AutoProcessor, BlipTextModel, BlipModel, BlipForImageTextRetrieval
from util import base64str_to_PILobj
import torch
import copy

class CustomBLIP(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 tokenizer=None,
                 config=None,
                 freeze_image_encoder=True,
                 freeze_text_encoder=True
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.num_labels = 1

        # self.blip = BlipModel()
        # self.blip = torch.load('model_output/blip_entire_model_kx_Salesforce-BlipForImageTextRetrieval-blip-itm-base-coco.pt')
        self.blip = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco")#BlipModel.from_pretrained("Salesforce/blip-vqa-base")
        # self.blip = torch.load('model_output/blip_entire_model_kx_Salesforce-BlipForImageTextRetrieval-blip-itm-base-coco-concat.pt')
        # BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        # BlipModel.from_pretrained("Salesforce/blip-vqa-base")
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # import pdb; pdb.set_trace()
        # self.text_encoder = copy.deepcopy(self.blip.text_encoder) # For custom retrain
        # self.vision_encoder = copy.deepcopy(self.blip.vision_encoder)
        # print(self.blip)
        # self.text_encoder = copy.deepcopy(self.blip.text_model) # For BlipModel
        
        self.text_encoder = copy.deepcopy(self.blip.text_encoder)
        
        self.vision_encoder = copy.deepcopy(self.blip.vision_model)

        if freeze_image_encoder:
            for _, p in self.vision_encoder.named_parameters():
                p.requires_grad_(False)

        if freeze_text_encoder:
            for _, p in self.text_encoder.named_parameters():
                p.requires_grad_(False)

        self.cross_entropy_loss = torch.nn.BCELoss(reduction='mean')

        self.cls_head = nn.Sequential(

            nn.Linear(self.text_encoder.config.hidden_size, 1024),#+self.vision_encoder.config.hidden_size
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(32, self.num_labels)

        )
        self.classifier = nn.Linear(self.text_encoder.config.hidden_size + self.vision_encoder.config.hidden_size, 1) 
        self.output_attentions = self.blip.config.output_attentions
        self.output_hidden_states = (
            self.blip.config.output_hidden_states
        )
        del self.blip       

    def forward(self, batch, device):

        pixel_values = batch['pixel_values'].to(device)
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # output_attentions = self.blip.config.output_attentions
        # output_hidden_states = (
        #     self.blip.config.output_hidden_states
        # )

        image_embeds = self.vision_encoder(pixel_values = pixel_values.squeeze(1),
                                           output_attentions=self.output_attentions,
                                           output_hidden_states=self.output_hidden_states,
                                           return_dict=True)[0]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        output = self.text_encoder(input_ids = input_ids.squeeze(1), 
                                    attention_mask=attention_mask.squeeze(1), 
                                    encoder_hidden_states=image_embeds, 
                                    encoder_attention_mask=image_atts, 
                                    return_dict=True)#.last_hidden_state[:, 0, :] # Return 
        # import pdb; pdb.set_trace()

        # combined_embeds = torch.cat((output.last_hidden_state[:, 0, :], image_embeds[:,0,:]), dim=1)
        # logits = self.classifier(combined_embeds)
        # prediction = torch.sigmoid(logits)

        prediction = self.cls_head(output.last_hidden_state[:, 0, :]) 
        prediction = torch.sigmoid(prediction)
            
        # logits = self.cls_head(combined_embeds)
        # prediction = torch.sigmoid(logits)
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
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-large-coco")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
       
        pixel_values = self.processor(images=base64str_to_PILobj(item["image"]).convert("RGB").resize((self.image_size, self.image_size)),
                                            return_tensors="pt")['pixel_values']
    
        text_output = self.processor(text = item['text'],
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

# blip_entire_model_kx_Salesforce-BlipForImageTextRetrieval-blip-itm-base-coco-new.pt
# CUR > 'blip_entire_model_kx_Salesforce-BlipForImageTextRetrieval-blip-itm-large-coco-new.pt'
