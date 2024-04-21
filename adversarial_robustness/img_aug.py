import random
from PIL import Image
import pandas as pd

import augly.image as imaugs
import augly.text as textaugs


def apply_aug(row):
    print(row['id'])
    try:
        image = Image.open(f"./hateful_memes/img/{row['id']}.png")

        aug = imaugs.Compose(
            [
                imaugs.Saturation(factor=2.0),
                imaugs.Scale(factor=0.6),
                imaugs.OverlayEmoji(emoji_path="./emojis/heart_eyes.jpg", emoji_size=0.1, x_pos=random.random(), y_pos=random.random(), opacity=0.4),
                imaugs.RandomNoise(mean=0.0,var=0.005)
            ]
        )
        aug_img = aug(image)
        aug_img.save(f"./augmented/augmented_{row['id']}.png")

        input_text = row['text']
        text_aug = textaugs.simulate_typos(input_text)

        row['aug_text'] = text_aug
        row['aug_img'] = f"augmented_{row['id']}.png"

    except FileNotFoundError as e:
        print(e)
        pass

    return row


if __name__ == '__main__':
    data_dir = './hateful_memes/'
    dataset_type = "dev_unseen" #test_unseen

    data = pd.read_json(path_or_buf=data_dir+dataset_type+".jsonl", lines=True)
    aug_data = data.apply(apply_aug,axis=1)

    with open(f"./hateful_memes/{dataset_type}.jsonl", "w") as f:
        f.write(aug_data.to_json(orient='records', lines=True, force_ascii=False).replace('\/', '/'))

    print("Generated augmented images")