import os
import torch
import clip
from PIL import Image

def scene_explanation_matching(recipe_explanations, img_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    result_list = []
    recipe_explanations = ["준비 과정"] + recipe_explanations

    # 감지된 scene의 프레임과 recipe_explanations를 CLIP 모델에 넣고
    # 가장 확률이 높은 단계를 매칭
    for scene_image_path in os.listdir(img_dir):
        image = preprocess(Image.open(scene_image_path)).unsqueeze(0).to(device)
        text = clip.tokenize(recipe_explanations).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # 각 scene이 몇 단계에 해당하는지 매칭
        result_list.append(max(probs) + 1)

    return result_list