# ------------------------------------------------------------------------------------
# Adapted from https://github.com/leeesangwon/bdd100k_to_VOC/blob/master/bdd_to_voc.py
# ------------------------------------------------------------------------------------
import argparse
import os
import os.path as osp
import shutil
import json
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom
from PIL import Image
from tqdm import tqdm

classes = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tramsform BDD100k dataset to VOC format"
    )
    parser.add_argument(
        "-i", "--input-dir", default="datasets/raw", help="raw datasets path"
    )
    parser.add_argument(
        "-o", "--output-dir", default="datasets/data", help="output path"
    )
    args = parser.parse_args()
    return args


def make_dir(path):
    if os.path.exists(path):
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))
    os.makedirs(path, exist_ok=True)


def bdd_to_voc(bdd_folder, save_prefix, split, time):
    """
    :param bdd_folder: a path to bdd100k which contains images, labels folder.
    :param xml_folder: a path to save the xml files.
    :param split: train / val
    :return:
    """
    image_path = bdd_folder + "/images/100k/%s"
    label_path = bdd_folder + "/labels/det_20/det_%s.json"

    image_folder = image_path % split
    json_path = label_path % split

    ann_dir = osp.join(save_prefix + split + "_" + time, "Annotations")
    make_dir(ann_dir)
    images_savepath = osp.join(save_prefix + split + "_" + time, "JPEGImages")
    make_dir(images_savepath)
    textfiles_savepath = os.path.join(
        save_prefix + split + "_" + time, "ImageSets", "Main"
    )
    make_dir(textfiles_savepath)

    split_files = []

    with open(json_path) as f:
        j = f.read()
    data = json.loads(j)

    for datum in tqdm(data):
        if datum["attributes"]["timeofday"] == time and "labels" in datum:
            annotation = Element("annotation")
            SubElement(annotation, "folder").text = split
            SubElement(annotation, "filename").text = datum["name"]
            size = get_size(osp.join(image_folder, datum["name"]))
            annotation.append(size)
            split_files.append(datum["name"][:-4])
            shutil.copyfile(
                osp.join(image_folder, datum["name"]),
                osp.join(images_savepath, datum["name"]),
            )
            # bounding box
            for label in datum["labels"]:
                if label["category"] in classes:
                    try:
                        box2d = label["box2d"]
                    except KeyError:
                        continue
                    else:
                        bndbox = get_bbox(box2d)

                    object_ = Element("object")
                    SubElement(object_, "name").text = label["category"]
                    SubElement(object_, "pose").text = "Unspecified"
                    SubElement(object_, "truncated").text = "0"
                    SubElement(object_, "difficult").text = "0"
                    object_.append(bndbox)
                    annotation.append(object_)

            xml_filename = osp.splitext(datum["name"])[0] + ".xml"
            with open(osp.join(ann_dir, xml_filename), "w") as f:
                f.write(prettify(annotation))
    split_files_wr = [x + "\n" for x in split_files]
    with open(os.path.join(textfiles_savepath, f"{split}.txt"), "w") as f:
        f.writelines(split_files_wr)


def get_size(image_path):
    i = Image.open(image_path)
    sz = Element("size")
    SubElement(sz, "width").text = str(i.width)
    SubElement(sz, "height").text = str(i.height)
    SubElement(sz, "depth").text = str(3)
    return sz


def get_bbox(box2d):
    bndbox = Element("bndbox")
    SubElement(bndbox, "xmin").text = str(round(box2d["x1"]))
    SubElement(bndbox, "ymin").text = str(round(box2d["y1"]))
    SubElement(bndbox, "xmax").text = str(round(box2d["x2"]))
    SubElement(bndbox, "ymax").text = str(round(box2d["y2"]))
    return bndbox


def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ElementTree.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")


def main():
    args = parse_args()
    bdd_dir = os.path.join(args.input_dir, "bdd100k")
    save_prefix = os.path.join(args.output_dir, "VOC2007_bdd")
    # 'daytime'
    for split in ["train", "val"]:
        print(f"Start converting BDD100k/{split}/daytime")
        bdd_to_voc(bdd_dir, save_prefix, split, "daytime")


if __name__ == "__main__":
    main()