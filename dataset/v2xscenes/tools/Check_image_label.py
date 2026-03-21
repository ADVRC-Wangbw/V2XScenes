from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import os

# ==================== UTILS ====================

def list_files_deep(path='.', suffix=('xml'), not_prefix=(('~', '.'))):
    files = []
    all_files = list(Path(path).glob('**/*.*'))
    
    if isinstance(suffix, str):
        suffix = suffix.lower()
    elif isinstance(suffix, tuple):
        suffix = tuple([x.lower() for x in suffix])

    for f in all_files:
        if f.is_file() and f.name.lower().endswith(suffix) and not f.name.startswith(not_prefix):
            files.append(f.resolve().as_posix())
    print(f'Total files: {len(files)}')
    return files


def rotate_point(x, y, alpha):
    rad = np.deg2rad(alpha)
    x_rot = x * np.cos(rad) - y * np.sin(rad)
    y_rot = x * np.sin(rad) + y * np.cos(rad)
    return x_rot, y_rot


def get_vehicle_color(v_type, cmap, types):
    cmap = plt.get_cmap(cmap)
    color = cmap(types.index(v_type) / len(types))
    return tuple(int(c * 255) for c in color[:3])


# ==================== MAIN PROCESSING ====================

def process_and_draw(json_path, img_path, save_path, file_name, cmap, types, line_thickness):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    img = cv2.imread(img_path)
    points, colors = [], []

    for obj in data:
        color = get_vehicle_color(obj['type'], cmap, types)
        
        r1 = obj['2d_box']['rect1']
        x1, y1, w1, h1 = int(r1['x']), int(r1['y']), int(r1['w']), int(r1['h'])
        
        r2 = obj['2d_box']['rect2']
        x2, y2, w2, h2 = int(r2['x']), int(r2['y']), int(r2['w']), int(r2['h'])

        pts = [(x1, y1), (x1 + w1, y1), (x1 + w1, y1 + h1), (x1, y1 + h1),
               (x2, y2), (x2 + w2, y2), (x2 + w2, y2 + h2), (x2, y2 + h2)]
        
        points.extend(pts)
        colors.append(color)

    points = np.array(points, dtype=np.int32)

    for i in range(0, len(points), 8):
        color = colors[i//8]
        p = points[i:i+8]

        cv2.polylines(img, [p[0:4]], True, color, line_thickness)
        cv2.polylines(img, [p[4:8]], True, color, line_thickness)
        
        cv2.line(img, tuple(p[0]), tuple(p[2]), color, line_thickness)
        cv2.line(img, tuple(p[1]), tuple(p[3]), color, line_thickness)

        for j in range(4):
            cv2.line(img, tuple(p[j]), tuple(p[j+4]), color, line_thickness)

    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, file_name)
    
    try:
        cv2.imwrite(full_path, img)
        print(f"Saved: {full_path}")
    except Exception as e:
        print(f"Error saving: {e}")


# ==================== MAIN ====================

def main():
    # Config
    file_name = '20240709_083937_300_1720485599_to_1720485650_51_173'
    ports_set = ['61','62', '63', '64', '65', '66', '67', '71', '72', '73', '74', '75', '76'] # Camera ports
    dataset_dir = "/home/myData/storage/code/HEAL/dataset/V2XScenes"
    
    # Visualization config
    cmap = 'rainbow'  # Colormap for vehicle types
    line_thickness = 2  # Line thickness for bounding boxes
    vehicle_types = ["Car", "Motorcycle", "Bicycle", "Pedestrian", "Truck", 
                     "Bus", "Trailer", "Van", 'Tricycle', 'autonomous vehicle', 'traffic cone']
    
    # Processing config
    frame_interval = 20  # Process every 5th frame (1 = all frames, 2 = every other, etc.)
    
    total = 0

    for ports in ports_set:
        jsons = list_files_deep(f'{dataset_dir}/{file_name}/label_new/road_camera/{ports}/', '.json')
        jpgs = list_files_deep(f'{dataset_dir}/{file_name}/road_camera/{ports}/', '.jpg')
        
        jpg_map = {Path(f).stem: f for f in jpgs}
        pairs = [(j, jpg_map[Path(j).stem]) for j in jsons if Path(j).stem in jpg_map]
        
        # Sort pairs by filename to ensure consistent ordering
        pairs.sort(key=lambda x: x[0])

        for idx, (j_path, i_path) in enumerate(pairs):
            if idx % frame_interval != 0:
                continue
                
            name = f"{Path(j_path).stem}.jpg"
            save = f'{dataset_dir}/{file_name}/visualization/vis_image/{ports}/'
            
            with open(j_path, "r") as f:
                total += len(json.load(f))
            
            process_and_draw(j_path, i_path, save, name, cmap, vehicle_types, line_thickness)

    print(f"\nDone! Total objects: {total}")


if __name__ == "__main__":
    main()