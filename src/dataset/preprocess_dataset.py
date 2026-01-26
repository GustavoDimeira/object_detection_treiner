import argparse, os, json, shutil
from pycocotools.coco import COCO
from src.dataset.data_aumentation.applyTiling import applyTiling
from PIL import Image

from src.utils.terminal_handler import LoadingBar



# ==========================
# Argument Parser
# ==========================

parser = argparse.ArgumentParser()

foldsHelp = "Define on how many folds your dataset will be splited (default: 5)"
valPercHelp = "Define the percentadge of images that will be used on validation (default: 0.3)"
tillingHelp = "Define in how many divisions the image will be segmentad (default: 1, no tilling)"

parser.add_argument('--folds', type=int, default=5, help=foldsHelp)
parser.add_argument('--valPerc', type=float, default=.3, help=valPercHelp)
parser.add_argument('--tilling', type=int, default=2, help=tillingHelp)

folds, valPerc, tilling = parser.parse_args().folds, parser.parse_args().valPerc, parser.parse_args().tilling

# =========================
# Load Dataset
# ========================

dir_root = os.path.dirname(__file__)

# remove previus folds, if exists
if (os.path.exists(os.path.join(dir_root, "folds"))):
    shutil.rmtree(os.path.join(dir_root, "folds"))

dataset_json = COCO(os.path.join(dir_root, "train/_annotations.coco.json"))

info = dataset_json.dataset["info"]
licenses = dataset_json.dataset["licenses"]
categories = dataset_json.dataset["categories"]
# subject to changes due to data augmentation
imagens = dataset_json.imgs
annotations = dataset_json.dataset["annotations"]

# =========================
# Data augmentation
# =========================

def save_imgs(imgs_data, imgs):
    for img, img_data in zip(imgs, imgs_data):
        name = img['file_name']

        img = Image.fromarray(img_data, 'RGB')
        save_path = os.path.join(
            os.path.abspath("./src/dataset/newDataSet"),
            name
        )

        img.save(save_path)

tiled_imgs, tiled_anns = [], []
bar = LoadingBar(len(imagens) - 1, "Tilling Images", ["Running preprocessing of tilling images", "this may take a while"])
bar.start()

for step, img_id in enumerate(imagens):
    bar.updateBar(step)

    img = dataset_json.loadImgs(img_id)[0]
    img_anns = dataset_json.loadAnns(dataset_json.getAnnIds(imgIds=[img_id]))

    new_imgs, new_anns, imgs_data = applyTiling(img, img_anns, tilling, len(tiled_imgs), len(tiled_anns))

    tiled_imgs.extend(new_imgs)
    tiled_anns.extend(new_anns)

    save_imgs(imgs_data, new_imgs)

# =========================
# =========================
# =========================

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2)
        
