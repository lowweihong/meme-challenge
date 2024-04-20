# HatefulMeme challenge


# Usage
## Extract raw dataset into Dataset object
1. Download hateful meme dataset zip file, unzip and placed into directory `hateful_memes`, run `python convert_dataset.py`

## Build new model and train
1. Specified your own model architeture, `train_*.py` file should be modified to use your own model architeture (minimal modification is expected).
2. Models architeture can be found in models/*.

## Performance
### CLIP
| Model    |  epo | head | map_dim | train_acc | dev_seen_acc | dev_seen_auc | dev_unseen_auc | Filename | batch_size | LR |
| -------- |  ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| openai/clip-vit-large-patch14  |    15 | concat | 32 |  - | - | 0.652 |0.760 |  clip_entire_model_added_sigmoid_gradclip.pt | 16 | 1e-4 | 
| openai/clip-vit-large-patch14  |    15 | concat | 1024 | - |0.674 | 0.772 | 0.7643 | clip_entire_model_added_sigmoid_gradclip.pt | 64 | 1e-4 |
| openai/clip-vit-large-patch14  |  20 | cross | 1024 | 0.9950 | 0.7 | 0.8278 | 0.811 | clip_entire_model_added_sigmoid_gradclip-cross.pt | 64 | 1e-4 |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K  |   20 | cross | 1024 | 0.98788 | 0.67 | 0.7594 | 0.745 | clip_entire_model_added_sigmoid_gradclip_laion-CLIP-ViT-B-32-laion2B-s34B-b79K-cross.pt | 64 | 1e-4 |


## BLIP
| Model    | epo | LR | train_acc| dev_seen_acc | dev_seen_auc | dev_unseen_auc | Filename | batch_size |
| -------- | -------  | ------- | ------- | ------- | ------- | ------- | ------- | ------- | 
| BlipModel "Salesforce/blip-image-captioning-large" |  10 | 5e-4 | | 0.578 | 0.6348 | 0.635 | blip_entire_model_kx_Salesforce-BlipModel-blip-image-captioning-large-inn.pt | 16 |
| BlipForImageTextRetrieval "Salesforce/blip-itm-large-coco" |  10 | 5e-4 |  | 0.528 | 0.6875 | 0.6718 | blip_entire_model_kx_Salesforce-BlipForImageTextRetrieval-blip-itm-large-coco-new.pt | 16 |
| BlipModel "Salesforce/blip-image-captioning-large" |  15 | 1e-3 |  | 0.558 | 0.6348 | 0.61619 | blip_entire_model_kx_Salesforce-BlipModel-blip-image-captioning-large-inn-LR-EPO.pt | 16 |
| BlipForImageTextRetrieval "Salesforce/blip-itm-large-coco" |  10 | 1e-3 | | 0.622 | 0.6495 | 0.6718 | blip_entire_model_kx_Salesforce-BlipForImageTextRetrieval-blip-itm-large-coco-new-LR-EPO.pt | 16 |
| BlipModel "Salesforce/blip-image-captioning-large" |  20 (max when 9) | 5e-4 | 0.800 | 0.612 | 0.6469 | 0.6379 | blip_entire_model_kx_Salesforce-BlipModel-blip-image-captioning-large-inn-LR-EPO.pt | 16 |
| BlipForImageTextRetrieval "Salesforce/blip-itm-large-coco" |  20 (max when 16) | 5e-4  | 0.7354 |0.546 | 0.6999 | 0.6908 | blip_entire_model_kx_Salesforce-BlipForImageTextRetrieval-blip-itm-large-coco-new-LR-EPO.pt | 16 |
| BlipModel "Salesforce/blip-image-captioning-large" CROSS |  20 (max when 19) | 5e-4 | 0.8003 | 0.612 | 0.6469 | 0.6289 | blip_entire_model_kx_Salesforce-BlipModel-blip-image-captioning-large-inn-cross.pt | 16 |
| Blip2Model "Salesforce/blip2-opt-2.7b" |  10 (max when 9) | 5e-4 | 0.8042 | 0.64 | 0.740 | 0.7266 | blip_entire_model_kx_Salesforce-BlipModel-blip2-inn-concat.pt | 8 |
| Blip2Model "Salesforce/blip2-flan-t5-xl" |  10 (max when 10) | 5e-4 | 0.81023 | 0.664 | 0.7371 | 0.7228 | blip_entire_model_kx_Salesforce-BlipModel-blip2-flan-t5-xlinn-concat.pt | 8 |
| Blip2Model "Salesforce/blip2-flan-t5-xl" |  10 (max when 5) | 5e-4 | 0.83482 | 0.662 | 0.7438 | 0.724 | blip_entire_model_kx_Salesforce-BlipModel-blip2-flan-t5-xlinn-concat-layer5.pt | 8 |
| Blip2Model "Salesforce/blip2-flan-t5-xl" |  10 (max when 10) | 5e-3 | 0.863 | 0.68 | 0.7318 | 0.72669 | blip_entire_model_kx_Salesforce-BlipModel-blip2-flan-t5-xlinn-concat-layer5-LR-5e-3.pt | 8 |
| Blip2Model "Salesforce/blip2-opt-2.7b" |  10 (max when 9) | 5e-4 | 0.779 | 0.628 | 0.7287 | 0.7155 | blip_entire_model_kx_Salesforce-BlipModel-blip2-inn-concat-epo30.pt | 8 |

- blip_entire_model_kx_Salesforce-BlipModel-blip-image-captioning-base-inn.pt *individual
- epo: 5 (15)
- batch_size: 16
- lr: 1e-4 (5e-4)
```
train AUC: 0.709
dev_seen AUC: 0.594
dev_unseen AUC: 0.583
{'dev_seen': 0.506,
 'dev_unseen': 0.6296296296296297,
 'train': 0.6448235294117647}
```

- blip_entire_model_kx_Salesforce-BlipForImageTextRetrieval-blip-itm-base-coco-new.pt
- epo: 20
- batch_size: 16
- lr: 4e-5
```
train AUC: 0.767
dev_seen AUC: 0.632
dev_unseen AUC: 0.627
{'dev_seen': 0.57,
 'dev_unseen': 0.6148148148148148,
 'train': 0.7216470588235294}
```

- blip_entire_model_kx_Salesforce-BlipForImageTextRetrieval-blip-itm-large-coco-new.pt
- epo: 15
- batch_size: 16
- lr: 5e-4
```

```
