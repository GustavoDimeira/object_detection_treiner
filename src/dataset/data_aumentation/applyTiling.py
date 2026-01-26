import PIL.Image as Image
import os, numpy as np, math

imgs_root_path = os.path.abspath("./src/dataset/train")

def applyTiling(img_obj, anns, tiling_ammount, next_img_id, next_ann_id):
    img = Image.open(os.path.join(imgs_root_path, img_obj['file_name']))
    img_format = img_obj['file_name'].split('.')[-1]

    original_width, original_height = img.size
    new_width, new_height = original_width // tiling_ammount, original_height // tiling_ammount

    new_anns = []
    new_imgs = []
    imgs_data = [] # will be returned in order to save the imagens, not overloading the memory

    for ann in anns:
        original_bbox = ann['bbox']
        
        x1 = math.floor(original_bbox[0])
        y1 = math.floor(original_bbox[1])
        x2 = math.floor(original_bbox[0] + original_bbox[2])
        y2 = math.floor(original_bbox[1] + original_bbox[3])

        x1_quadrant = int(x1 / new_width)
        y1_quadrant = int(y1 / new_height)
        x2_quadrant = int(x2 / new_width)
        y2_quadrant = int(y2 / new_height)

        if (x1_quadrant == x2_quadrant and y1_quadrant == y2_quadrant): # all in the same tile
            image_id = y1_quadrant * tiling_ammount + x1_quadrant + next_img_id
            bbox = [
                x1 - x1_quadrant * new_width, # new_x
                y1 - y1_quadrant * new_height, # new_y
                x2 - x1, # width
                y2 - y1, # height
            ]
            area = (x2 - x1) * (y2 - y1)

            new_anns.append({
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": ann['category_id'],
                "bbox": bbox,
                "area": area,
                "iscrowd": ann['iscrowd']
            })

            next_ann_id += 1

    img = np.array(img)

    for i in range(tiling_ammount):
        for j in range(tiling_ammount):
            new_tile = img[
                new_height * i: new_height * (i + 1),
                new_width * j: new_width * (j + 1),
                :
            ]
   
            imgs_data.append(new_tile)
            new_imgs.append(
                {
                    "id": next_img_id,
                    "license": 1,
                    "file_name": f"{img_obj['file_name'].split('.')[0]}_tile_{i}x{j}.{img_format}",
                    "height": new_height,
                    "width": new_width,
                },
            )
            next_img_id += 1
            
    return new_imgs, new_anns, imgs_data
