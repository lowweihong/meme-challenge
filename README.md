# HatefulMeme challenge


# Usage
## Extract raw dataset into Dataset object
1. Download hateful meme dataset zip file, unzip and place into directory `hateful_memes`, run `python convert_dataset.py`
2. 
## Extract Adversarial/Image Augmented Datasets
   1. Download adversarial dataset zip from external link [here](https://drive.google.com/file/d/11DTJKmRSW8fKXxgbtqwDWEmLAAA2beyg/view?usp=drive_link) (185MB) and augmented dataset zip [here](https://drive.google.com/file/d/1-1eYLaY6-jjFl0weE6MnixEeqISqwGwS/view?usp=drive_link)(2.77GB).
   2. Unzip files into directory `hateful_memes`
   3. Copy desired dataset jsonl files from `adversarial_robustness/adv_datasets` and place it into `hateful_memes`
   4. Run `python convert_adv_dataset.py`

## Build new model and train
1. Specified your own model architeture, `train_*.py` file should be modified to use your own model architecture (minimal modification is expected).
2. Models architecture can be found in models/*.


## Performance
### CLIP
|            CLIP Variations            | #Proj layer | map_dim | #P.O layer | Epo | BS |    LR    |  Fusion  | Train ACC |    AUC   |            |           |             | ACC      |            |           |             | Model FP                                                                                |
|:-------------------------------------:|-------------|:-------:|:----------:|:---:|:--:|:--------:|:--------:|:---------:|:--------:|:----------:|:---------:|:-----------:|----------|------------|-----------|-------------|-----------------------------------------------------------------------------------------|
|                                       |             |         |            |     |    |          |          |           | Dev seen | Dev unseen | Test seen | Test unseen | Dev seen | Dev unseen | Test seen | Test unseen |                                                                                         |
| clip-vit-large-patch14                | 1           | 1024    | 1          | 15  | 64 | 1.00E-04 | concat   |     | 0.7658   | 0.755      | 0.7643    | 0.797       | 69.00%   | 70.20%     | 73.10%    | 73.30%      | clip_entire_model_added_sigmoid_gradclip.pt                                             |
| clip-vit-large-patch14                | 5           | 1024    | 1          | 20  | 64 | 1.00E-04 | cross    | 99.50%    | 0.8276   | 0.8113     | 0.834     | 0.824       | 69.60%   | 75.70%     | 72.90%    | 75.60%      | clip_entire_model_added_sigmoid_gradclip-cross.pt                                       |
| openai/clip-vit-large-patch14         | 5           | 1024    | 1          | 20  | 64 | 1.00E-04 | self-att | 64.48%   | 0.4927   | 0.5291     | 0.505     | 0.51        | 50.60%   | 63.00%     | 51.00%    | 63.50%      | clip_entire_model_added_sigmoid_gradclip-att-layer5.pt                                  |
| openai/clip-vit-large-patch14         | 10          | 1024    | 1          | 20  | 64 | 1.00E-04 | cross    | 97.35%    | 781      | 0.765      | 0.785     | 0.766       | 69.20%   | 73.90%     | 69.60%    | 73.30%      | clip_entire_model_added_sigmoid_gradclip-cross-layer10.pt                               |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K | 5           | 1024    | 1          | 20  | 64 | 1.00E-04 | cross    | 98.79%    | 0.7539   | 0.7381     | 0.769     | 0.766       | 67.00%   | 69.10%     | 69.10%    | 73.30%      | clip_entire_model_added_sigmoid_gradclip_laion-CLIP-ViT-B-32-laion2B-s34B-b79K-cross.pt |


## DRAFT
### CLIP
| Model    |  epo | head | map_dim | train_acc | dev_seen_acc | dev_seen_auc | dev_unseen_auc | Filename | batch_size | LR | layer |
| -------- |  ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |------- |
| openai/clip-vit-large-patch14  |    15 | concat | 32 |  - | - | 0.652 |0.760 |  clip_entire_model_added_sigmoid_gradclip.pt | 16 | 1e-4 | 1 |
| openai/clip-vit-large-patch14  |    15 | concat | 1024 | - |0.674 | 0.772 | 0.7643 | clip_entire_model_added_sigmoid_gradclip.pt | 64 | 1e-4 | 1 |
| openai/clip-vit-large-patch14  |  20 | cross | 1024 | 0.9950 | 0.7 | 0.8278 | 0.811 | clip_entire_model_added_sigmoid_gradclip-cross.pt | 64 | 1e-4 | 5 |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K  |   20 | cross | 1024 | 0.98788 | 0.67 | 0.7594 | 0.745 | clip_entire_model_added_sigmoid_gradclip_laion-CLIP-ViT-B-32-laion2B-s34B-b79K-cross.pt | 64 | 1e-4 | 5 |
| openai/clip-vit-large-patch14  |  20 | self-att | 1024 | 0.6448 | 0.506 | 0.538 | 0.5145 | clip_entire_model_added_sigmoid_gradclip-att-layer5.pt | 64 | 1e-4 | 5 |
| openai/clip-vit-large-patch14  |  20 (max when epo 6) | cross | 1024 | 0.97352 | 0.688 | 0.7825 | 0.76997 | clip_entire_model_added_sigmoid_gradclip-cross-layer10.pt | 64 | 1e-4 | 10 |
| openai/clip-vit-large-patch14  |  20 | cross | 1024 | 0.6504 | 0.504 | 0.54513 |  | clip_entire_model_added_sigmoid_gradclip-cross-unfreeze-last-block.pt | 8 | 1e-4 | 5 |


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
| Dino+BGE "facebook/dinov2-large" + BAAI/bge-m3 |  20 | 5e-4 | 0.9815 | 0.59 | 0.6067 | 0.622 | dino_large_bge.pt | 32 |

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
