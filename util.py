import base64
from PIL import Image
import io
import torch

def evaluate(model, dataloader, device='cpu'):
    correct_normal = 0
    total = 0
    model = model.to(device)
    
    with torch.no_grad():
        for batch in dataloader:
            labels = batch['labels'].to(device)
            labels = labels.reshape(-1,1)
            labels = labels.to(device)

            output = model(batch, device)
            predicted = torch.as_tensor((output - 0.5) > 0, dtype=torch.int32)

            total += labels.size(0)
            correct_normal += (predicted==labels).sum().item()


        accuracy = correct_normal/total
    return accuracy


def base64str_to_PILobj(base64_string):
    '''
    Args
    - base64_string (str): based64 encoded representing an image

    Output
    - PIL object (use .show() to display)
    '''
    image_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(image_data))
    return img