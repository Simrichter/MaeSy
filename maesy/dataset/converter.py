import json
import os

from tqdm import tqdm


def datumaro_to_devils_yolo(datumaro_dir: str):
        """
        Convert Image annotations exported from cvat in datumaro JSON format into DevilsYolo format.
        Lines are represented by their two endpoint coordinates in xyxy format.
        All coordinates are normalized

        Args:
            :param datumaro_dir: Path to the datumaro dataset root folder (the one that contains the annotations folder and the images folder)
        """

        print("="*60)
        print(f"Converting datumaro to DevilsYolo format")
        print(f"Dataset root path: {datumaro_dir}")
        print("=" * 60)

        def _fix_multipoint_line(datadict):
            for (img_i, img_dat) in enumerate(datadict['items']):
                anns_to_be_added = []
                for ann_i, annot in enumerate(img_dat['annotations']):
                    if annot['type'] == "polyline":
                        points = annot['points']
                        if len(points) != 4:
                            print(f"Fixing Polyline for img: {img_dat['id']} with {len(points) / 2 - 1} lines")
                            for i in range(0, len(points) - 3, 2):
                                p1 = (points[i], points[i + 1])
                                p2 = (points[i + 2], points[i + 3])
                                new_ann = annot.copy()
                                new_ann["points"] = [p1[0], p1[1], p2[0], p2[1]]
                                if not (p1[0] == p2[0] and p1[1] == p2[1]):  # Check for zero-length vector
                                    anns_to_be_added.append(new_ann)
                                else:
                                    print(
                                        f"Skipping double point for annotation {ann_i} in file {img_dat['id']} ({img_i}) due to zero-length vector: {p1} and {p2}")

                            annot['points'] = anns_to_be_added[0]['points']
                            anns_to_be_added = anns_to_be_added[1:]
                img_dat['annotations'].extend(anns_to_be_added)

        with open(f"{datumaro_dir}/annotations/default.json", 'r') as f:
            data = json.load(f)
        print("JSON loaded successfully")
        print("Checking for multi-point line annotations and fixing them if necessary...")
        _fix_multipoint_line(data)

        img_dir = f"{datumaro_dir}/images/default"

        print("Writing .txt annotations...")
        for img_ind, img_data in enumerate(tqdm(data['items'])):
            with (open(os.path.join(img_dir, img_data['id'] + ".txt"), 'w') as f):
                normalize = img_data["image"]["size"]
                normalize.reverse() # datumaro saves height, width
                for ann_ind, ann in enumerate(img_data['annotations']):
                    if ann['type'] == "bbox":
                        cx = ann['bbox'][0] / normalize[0]
                        cy = ann['bbox'][1] / normalize[1]
                        w = ann['bbox'][2] / normalize[0]
                        h = ann['bbox'][3] / normalize[1]
                        cx += 0.5 * w
                        cy += 0.5 * h
                        f.write(f"{ann['label_id']} {cx} {cy} {w} {h}\n")
                    elif ann['type'] == "polyline":
                        p = ann['points']
                        if len(p) != 4:
                            print(
                                f"Warning: Skipping annotation in file {img_data['id']} due to incorrect number of points: {p}")
                            continue
                        f.write(f"{ann['label_id']} {p[0]/normalize[0]} {p[1]/normalize[1]} {p[2]/normalize[0]} {p[3]/normalize[1]}\n")
        print("Conversion finished successfully!")