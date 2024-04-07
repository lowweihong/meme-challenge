from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import BlipForConditionalGeneration, BlipForQuestionAnswering, AutoProcessor, BlipTextModel, BlipModel, BlipForImageTextRetrieval
from util import base64str_to_PILobj
import torch

class CustomBLIP(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 tokenizer=None,
                 config=None,
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.num_labels = 1

        self.model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        for param in self.model.parameters():
            param.requires_grad = False

        #self.text_encoder = self.model.text_model
        self.text_encoder = self.model.text_encoder
        
        self.vision_encoder = self.model.vision_model
        self.cross_entropy_loss = torch.nn.BCELoss(reduction='mean')

        self.cls_head = nn.Sequential(

            nn.Linear(self.text_encoder.config.hidden_size, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, self.num_labels)

        )        

    def forward(self, batch, device):

        pixel_values = batch['pixel_values'].to(device)
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        image_embeds = self.vision_encoder(pixel_values.squeeze(1))[0]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        output = self.text_encoder(input_ids = input_ids.squeeze(1), 
                                    attention_mask=attention_mask.squeeze(1), 
                                    encoder_hidden_states=image_embeds, 
                                    encoder_attention_mask=image_atts, 
                                    return_dict=True)


        prediction = self.cls_head(output.last_hidden_state[:, 0, :]) 
        prediction = torch.sigmoid(prediction)
            
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
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

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