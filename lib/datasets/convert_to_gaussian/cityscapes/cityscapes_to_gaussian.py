import json
import os

import lib.datasets.convert_to_gaussian.cityscapes.cityscapesscripts.evaluation.instances2dict_with_polygons as cs
import lib.utils.boxes as bboxs_util
import lib.utils.segms as segms_util


def convert_cityscapes(data_dir, out_dir):
    '''
    convert from cityscapes format to gaussian format
    :param data_dir:
    :param out_dir:
    :return:
    '''

    sets = ['gtFine_val']
    ann_dirs = ['gtFine_trainvaltest/gtFine/val']
    json_name = 'instancesonly_filtered_%s.json'
    ends_in = '%s_polygons.json'
    img_id = 0
    ann_id = 0
    cat_id = 1
    category_dict={}

    category_list=[
        'person',
        'rider',
        'car',
        'truck',
        'bus',
        'motorcycle',
        'bicycle',
        'road',
        'sidewalk',
        'building',
        'wall',
        'fence',
        'pole',
        'traffic sign',
        'traffic light',
        'vegetation',
        'terrain',
        'sky'
    ]

    for data_set, ann_dir in zip(sets, ann_dirs):
        print('Starting %s' % data_set)
        ann_dict = {}
        images = []
        annotations = []
        ann_dir = os.path.join(data_dir, ann_dir)
        for root, _, files in os.walk(ann_dir):
            for filename in files:
                if filename.endswith(ends_in % data_set.split('_')[0]):
                    if len(images) % 50 == 0:
                        print("Processed %s images, %s annotations" % (
                            len(images), len(annotations)))
                    json_ann = json.load(open(os.path.join(root, filename)))
                    image = {}
                    image['id'] = img_id
                    img_id += 1

                    image['width'] = json_ann['imgWidth']
                    image['height'] = json_ann['imgHeight']
                    image['file_name'] = filename[:-len(
                        ends_in % data_set.split('_')[0])] + 'leftImg8bit.png'
                    image['seg_file_name'] = filename[:-len(
                        ends_in % data_set.split('_')[0])] + \
                        '%s_labelIds.png' % data_set.split('_')[0]
                    images.append(image)

                    fullname = os.path.join(root, image['seg_file_name'])
                    objects = cs.instances2dict_with_polygons(
                        [fullname], verbose=False)[fullname]

                    for object_cls in objects:
                        if object_cls not in category_list:
                            continue  # skip non-instance categories

                        for obj in objects[object_cls]:
                            if obj['contours'] == []:
                                print('Warning: empty contours.')
                                continue  # skip non-instance categories

                            len_p = [len(p) for p in obj['contours']]
                            if min(len_p) <= 4:
                                print('Warning: invalid contours.')
                                continue  # skip non-instance categories

                            ann = {}
                            ann['id'] = ann_id
                            ann_id += 1
                            ann['image_id'] = image['id']
                            ann['segmentation'] = obj['contours']

                            if object_cls not in category_dict:
                                category_dict[object_cls] = cat_id
                                cat_id += 1
                            ann['category_id'] = category_dict[object_cls]
                            ann['iscrowd'] = 0
                            ann['area'] = obj['pixelCount']
                            ann['bbox'] = bboxs_util.xyxy_to_xywh(
                                segms_util.polys_to_boxes(
                                    [ann['segmentation']])).tolist()[0]

                            annotations.append(ann)

        ann_dict['images'] = images
        categories = [{"id": category_dict[name], "name": name} for name in
                      category_dict]
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        with open(os.path.join(out_dir, json_name % data_set), 'w') as outfile:
            outfile.write(json.dumps(ann_dict))


if __name__ == '__main__':
    data_dir = '/media/pesong/e/dl_gaussian/data/cityscapes'
    out_dir = './'
    convert_cityscapes(data_dir, out_dir)