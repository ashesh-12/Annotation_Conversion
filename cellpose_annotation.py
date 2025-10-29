import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pprint

# ------------------- Converter ------------------- #
class GeneralCellposeConverter:
    """
    Convert XML annotations (polygons, boxes, points, lines, ellipses, etc.)
    to Cellpose-ready .npy masks.
    """

    def __init__(self, xml_path: str, save_folder: str = "npy_file"):
        self.xml_path = xml_path
        self.save_folder = save_folder
        os.makedirs(save_folder, exist_ok=True)

        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()

    def _parse_points(self, points_str):
        return [tuple(map(float, p.split(','))) for p in points_str.split(';')]

    def _create_mask(self, width, height, objects):
        mask = np.zeros((height, width), dtype=np.uint16)
        for idx, obj in enumerate(objects, start=1):
            shape_type = obj['type']
            shape_data = obj['data']
            img = Image.new('I', (width, height), 0)
            draw = ImageDraw.Draw(img)

            if shape_type == 'polygon':
                draw.polygon(shape_data, outline=idx, fill=idx)
            elif shape_type == 'box':
                draw.rectangle(shape_data, outline=idx, fill=idx)
            elif shape_type == 'polyline':
                draw.line(shape_data, fill=idx, width=3)
            elif shape_type == 'point':
                x, y = shape_data
                draw.ellipse([x-1, y-1, x+1, y+1], outline=idx, fill=idx)
            elif shape_type == 'ellipse':
                draw.ellipse(shape_data, outline=idx, fill=idx)

            mask += np.array(img, dtype=np.uint16)
        return mask

    def run(self):
        for image_tag in self.root.findall('image'):
            image_name = image_tag.attrib['name']
            width = int(image_tag.attrib['width'])
            height = int(image_tag.attrib['height'])

            objects = []

            # Polygons
            for poly in image_tag.findall('polygon'):
                points = self._parse_points(poly.attrib['points'])
                objects.append({'type':'polygon', 'data': points})

            # Boxes
            for box in image_tag.findall('box'):
                xtl = float(box.attrib['xtl'])
                ytl = float(box.attrib['ytl'])
                xbr = float(box.attrib['xbr'])
                ybr = float(box.attrib['ybr'])
                objects.append({'type':'box', 'data': [(xtl, ytl), (xbr, ybr)]})

            # Polylines
            for line in image_tag.findall('polyline'):
                points = self._parse_points(line.attrib['points'])
                objects.append({'type':'polyline', 'data': points})

            # Points
            for pt in image_tag.findall('points'):
                points = self._parse_points(pt.attrib['points'])
                for p in points:
                    objects.append({'type':'point', 'data': p})

            # Ellipses
            for ellipse in image_tag.findall('ellipse'):
                cx = float(ellipse.attrib['cx'])
                cy = float(ellipse.attrib['cy'])
                rx = float(ellipse.attrib['rx'])
                ry = float(ellipse.attrib['ry'])
                bbox = [cx-rx, cy-ry, cx+rx, cy+ry]
                objects.append({'type':'ellipse', 'data': bbox})

            mask = self._create_mask(width, height, objects)
            npy_path = os.path.join(self.save_folder, os.path.splitext(image_name)[0] + '.npy')
            np.save(npy_path, mask, allow_pickle=True)
            print(f"Saved mask: {npy_path}")

# ------------------- Loader ------------------- #
class CellposeMaskLoader:
    """
    Load Cellpose-ready .npy masks and provide inspection & visualization.
    """

    def __init__(self, npy_folder="npy_file"):
        self.npy_folder = npy_folder
        os.makedirs(npy_folder, exist_ok=True)
        self.files = [f for f in os.listdir(npy_folder) if f.endswith('.npy')]
        self.masks = {}

    def load(self, filename):
        path = os.path.join(self.npy_folder, filename)
        mask = np.load(path, allow_pickle=True)
        self.masks[filename] = mask
        print(f"✅ Loaded mask: {filename} | Shape: {mask.shape} | Dtype: {mask.dtype}")
        return mask

    def summary(self):
        print("\n--- Mask Summary ---")
        for f in self.files:
            path = os.path.join(self.npy_folder, f)
            mask = np.load(path, allow_pickle=True)
            unique_labels = np.unique(mask)
            print(f"{f}: shape={mask.shape}, unique_labels={unique_labels}")

    def print_full_data(self):
        print("\n--- Full Mask Data ---")
        for f in self.files:
            path = os.path.join(self.npy_folder, f)
            mask = np.load(path, allow_pickle=True)
            print(f"\nFile: {f} | Shape: {mask.shape} | Dtype: {mask.dtype}")
            pprint.pprint(mask)

    def visualize(self, filename):
        if filename not in self.masks:
            mask = self.load(filename)
        else:
            mask = self.masks[filename]

        plt.imshow(mask, cmap='jet')
        plt.title(f"Mask: {filename}")
        plt.axis('off')
        plt.show()


# ------------------- Example Usage ------------------- #
if __name__ == "__main__":
    xml_path = r"D:\MISSION X\Annotation_Conversion\annotations.xml"
    save_folder = r"D:\MISSION X\Annotation_Conversion\npy_file"  # <-- NEW folder

    # Convert XML → NPY
    converter = GeneralCellposeConverter(xml_path, save_folder)
    converter.run()

    # Load and inspect
    loader = CellposeMaskLoader(save_folder)
    loader.summary()
    loader.print_full_data()
    if loader.files:
        loader.visualize(loader.files[0])
