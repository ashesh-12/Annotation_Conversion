import numpy as np
import xml.etree.ElementTree as ET
from pprint import pprint

class GenericAnnotationConverter:
    """Convert any CVAT-style XML annotations to NPZ dynamically (all shapes)."""

    def __init__(self, xml_path, save_path="annotations.npz"):
        self.xml_path = xml_path
        self.save_path = save_path
        self.data = {}

    def parse(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        for image in root.findall("image"):
            image_name = image.get("name")
            if "image_names" not in self.data:
                self.data["image_names"] = []
            self.data["image_names"].append(image_name)

            for shape in image:
                tag = shape.tag.lower()
                if tag not in self.data:
                    self.data[tag] = []

                # --- Generic handling ---
                if tag in ["box", "ellipse"]:
                    # Convert attributes to float list
                    values = [float(shape.get(k)) for k in shape.keys()]
                    self.data[tag].append(values)
                elif tag in ["polygon", "polyline", "points"]:
                    pts = np.array([list(map(float, p.split(','))) for p in shape.get("points").split(';')], dtype=np.float32)
                    self.data[tag].append(pts)
                elif tag in ["cuboid"]:
                    coords = [float(shape.get(k)) for k in shape.keys() if shape.get(k)]
                    self.data[tag].append(coords)
                elif tag in ["skeleton", "mask"]:
                    self.data[tag].append(shape.attrib)
                elif tag == "tag":
                    self.data[tag].append(shape.get("label"))
                else:
                    # Any other unknown shape type
                    self.data[tag].append(shape.attrib)

    def save(self):
        """Save all collected shapes into a compressed NPZ."""
        npz_dict = {}
        for key, value in self.data.items():
            if len(value) == 0:
                npz_dict[key] = np.array([], dtype=object)
            else:
                npz_dict[key] = np.array(value, dtype=object)
        np.savez_compressed(self.save_path, **npz_dict)
        print(f"✅ Saved all annotations to {self.save_path}")

    def run(self):
        self.parse()
        self.save()

class GenericAnnotationLoader:
    """Load NPZ annotations dynamically and inspect any shape type."""

    def __init__(self, npz_path):
        self.npz_path = npz_path
        self.data = self.load_npz()

    def load_npz(self):
        npz_data = np.load(self.npz_path, allow_pickle=True)
        print(f"✅ Loaded annotation data from: {self.npz_path}")
        print("Available keys:", list(npz_data.keys()))
        return {key: npz_data[key] for key in npz_data.keys()}

    def summary(self):
        print("\nAnnotation Summary:")
        for key, value in self.data.items():
            print(f"{key}: {len(value)} items")

    def preview(self, key, n=2):
        """Pretty print first n items of a given annotation type."""
        if key not in self.data:
            print(f"❌ Key '{key}' not found.")
            return

        print(f"\nPreview of {key} (first {n} items):")
        for i, item in enumerate(self.data[key][:n]):
            print(f"[{i}] {key}:")
            # Handle numpy arrays (polygons, points, etc.)
            if isinstance(item, np.ndarray):
                if item.ndim == 1:  # 1D array
                    print(f"  {item}")
                else:  # 2D array, e.g., polygons
                    for point in item:
                        print(f"   x={point[0]:.2f}, y={point[1]:.2f}")
            # Handle dictionaries (masks, skeletons)
            elif isinstance(item, dict):
                pprint(item, indent=4)
            # Handle simple lists or strings (tags, cuboids, boxes)
            else:
                print(f"  {item}")

    def get(self, key):
        """Return full data of a specific annotation type."""
        return self.data.get(key, None)


# ------------------ Example Usage ------------------ #

if __name__ == "__main__":
    # Convert XML → NPZ
    converter = GenericAnnotationConverter(r"C:\Users\User\Desktop\WY\annotations.xml")
    converter.run()

    # Load NPZ
    loader = GenericAnnotationLoader("annotations.npz")
    loader.summary()
    loader.preview("polygon")
    loader.preview("mask")
    loader.preview("box")
