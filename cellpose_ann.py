import os
import numpy as np
import xml.etree.ElementTree as ET
from pprint import pprint


class GenericAnnotationConverter:
    """Convert any CVAT-style XML annotations to individual NPY files per image."""

    def __init__(self, xml_path, save_folder="npy_files"):
        self.xml_path = xml_path
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

    def parse(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        for image in root.findall("image"):
            image_name = image.get("name")
            image_id = os.path.splitext(os.path.basename(image_name))[0]  # remove extension
            image_data = {"image_name": image_name}

            for shape in image:
                tag = shape.tag.lower()
                if tag not in image_data:
                    image_data[tag] = []

                # --- Generic handling ---
                if tag in ["box", "ellipse"]:
                    values = [float(shape.get(k)) for k in shape.keys() if shape.get(k) is not None]
                    image_data[tag].append(values)
                elif tag in ["polygon", "polyline", "points"]:
                    pts = np.array([list(map(float, p.split(','))) for p in shape.get("points").split(';')], dtype=np.float32)
                    image_data[tag].append(pts)
                elif tag in ["cuboid"]:
                    coords = [float(shape.get(k)) for k in shape.keys() if shape.get(k)]
                    image_data[tag].append(coords)
                elif tag in ["skeleton", "mask"]:
                    image_data[tag].append(shape.attrib)
                elif tag == "tag":
                    image_data[tag].append(shape.get("label"))
                else:
                    image_data[tag].append(shape.attrib)

            # Save one .npy file per image
            save_path = os.path.join(self.save_folder, f"{image_id}.npy")
            np.save(save_path, image_data, allow_pickle=True)
            print(f"✅ Saved {image_name} → {save_path}")

    def run(self):
        self.parse()


class GenericAnnotationLoader:
    """Load individual NPY annotation files and inspect any image."""

    def __init__(self, folder_path="npy_files"):
        self.folder_path = folder_path

    def list_files(self):
        files = [f for f in os.listdir(self.folder_path) if f.endswith(".npy")]
        print(f"Found {len(files)} NPY files:")
        for f in files:
            print(f" - {f}")
        return files

    def load(self, image_name):
        """Load annotation for a specific image (without extension)."""
        npy_path = os.path.join(self.folder_path, f"{image_name}.npy")
        if not os.path.exists(npy_path):
            print(f"❌ File not found: {npy_path}")
            return None
        data = np.load(npy_path, allow_pickle=True).item()
        print(f"✅ Loaded {npy_path}")
        return data

    def preview(self, image_name, key=None, n=2):
        """Preview annotation contents for a specific image."""
        data = self.load(image_name)
        if data is None:
            return
        print(f"\n🖼️ Image: {data['image_name']}")
        for k, v in data.items():
            if k == "image_name":
                continue
            print(f"{k}: {len(v)} items")
            if key and k != key:
                continue
            for i, item in enumerate(v[:n]):
                print(f"  [{i}] {k}:")
                if isinstance(item, np.ndarray):
                    if item.ndim == 1:
                        print(f"    {item}")
                    else:
                        for point in item:
                            print(f"     x={point[0]:.2f}, y={point[1]:.2f}")
                elif isinstance(item, dict):
                    pprint(item, indent=4)
                else:
                    print(f"    {item}")


# ------------------ Example Usage ------------------ #
if __name__ == "__main__":
    # Convert XML → Individual NPYs
    converter = GenericAnnotationConverter(r"D:\MISSION X\Annotation_Conversion\annotations.xml",save_folder="npy_files")
    converter.run()

    # Load and inspect
    loader = GenericAnnotationLoader("npy_files")
    loader.list_files()
    loader.preview("BH-07A-40X")       # Replace 'image_1' with actual image name (without extension)
