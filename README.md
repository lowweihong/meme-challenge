# HatefulMeme challenge


# Usage
## Extract raw dataset into Dataset object
1. Download hateful meme dataset zip file, unzip and place into directory `hateful_memes`, run `python convert_dataset.py`

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
| clip-vit-large-patch14                | 1           | 1024    | 1          | 10  | 64 | 1.00E-04 | concat   |  99.08%   | 0.7658   | 0.755      | 0.7643    | 0.797       | 69.00%   | 70.20%     | **73.10%**    | 73.30%      | clip_entire_model_added_sigmoid_gradclip.pt                                             |
| clip-vit-large-patch14                | 5           | 1024    | 1          | 20  | 64 | 1.00E-04 | cross    | **99.50%**    | **0.8276**   | **0.8113** | **0.834**     | **0.824**       | 69.60%   | **75.70%**     | 72.90%    | **75.60%**      | clip_entire_model_added_sigmoid_gradclip-cross.pt                                       |
| openai/clip-vit-large-patch14         | 5           | 1024    | 1          | 20  | 64 | 1.00E-04 | self-att | 64.48%   | 0.4927   | 0.5291     | 0.505     | 0.51        | 50.60%   | 63.00%     | 51.00%    | 63.50%      | clip_entire_model_added_sigmoid_gradclip-att-layer5.pt                                  |
| openai/clip-vit-large-patch14         | 10          | 1024    | 1          | 20  | 64 | 1.00E-04 | cross    | 97.35%    | 0.781      | 0.765      | 0.785     | 0.766       | 69.20%   | 73.90%     | 69.60%    | 73.30%      | clip_entire_model_added_sigmoid_gradclip-cross-layer10.pt                               |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K | 5           | 1024    | 1          | 20  | 64 | 1.00E-04 | cross    | 98.79%    | 0.7539   | 0.7381     | 0.769     | 0.766       | **67.00%**   | 69.10%     | 69.10%    | 73.30%      | clip_entire_model_added_sigmoid_gradclip_laion-CLIP-ViT-B-32-laion2B-s34B-b79K-cross.pt |

### BLIP
| Variations | Pretrained model used                  | #Proj layer | #P.O layer | Epo | BS |    LR    |              Fusion             | Train ACC |    AUC   |            |           |             |    ACC   |            |           |             | Model FP                                                                            |
|------------|----------------------------------------|-------------|:----------:|:---:|:--:|:--------:|:-------------------------------:|:---------:|:--------:|:----------:|:---------:|:-----------:|:--------:|:----------:|:---------:|:-----------:|-------------------------------------------------------------------------------------|
|            |                                        |             |            |     |    |          |                                 |           | Dev seen | Dev unseen | Test seen | Test unseen | Dev seen | Dev unseen | Test seen | Test unseen |                                                                                     |
| BLIP ITM   | Salesforce/blip-itm-large-coco         | 0           |          6 |  20 | 16 | 5.00E-04 | Cross att between image & text  |    0.7354 | 0.6998   | 0.691      | 0.6891    | 0.679       | 54.60%   | 64.60%     | 56.50%    | 65.20%      | blip_entire_model_Salesforce-BlipForImageTextRetrieval-blip-itm-large-coco-new.pt   |
| BLIP Base  | Salesforce/blip-image-captioning-large | 5           |          1 |  20 | 16 | 5.00E-04 | concat                          |       0.8 | 0.6485   | 0.6379     | 0.6574    | 0.678       | 59.20%   | 64.10%     | 60.30%    | 66.10%      | blip_entire_model_Salesforce-BlipModel-blip-image-captioning-large-inn-LR-EPO.pt |
| BLIP2      | Salesforce/blip2-opt-2.7b              | 5           |          1 |  10 |  8 | 5.00E-04 | concat                          |    0.8042 | 0.74     | 0.726      | 0.745     | 0.745       | 64.00%   | 69.10%     | 65.20%    | 69.80%      | blip_entire_model_Salesforce-BlipModel-blip2-inn-concat.pt                          |
| BLIP2      | Salesforce/blip2-flan-t5-xl            | 5           |          1 |  10 |  8 | 5.00E-03 | Concat image+text+qformer layer |     0.863 | 0.75     | 0.733      | 0.7682    | 0.764       | 68.00%   | 70.90%     | 68.80%    | 71.60%      | blip_entire_model_Salesforce-BlipModel-blip2-flan-t5-xlinn-concat-layer5-LR-5e-3.pt |
|            |                                        |             |            |     |    |          |                                 |           |          |            |           |             |          |            |           |             |                                                                                     |