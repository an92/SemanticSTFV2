import os
import numpy as np

color_map = {
    0: [245, 150, 100],     # car
    1: [245, 230, 100],     # bicycle
    2: [150, 60, 30],       # motorcycle
    3: [180, 30, 80],       # truck
    4: [255, 0, 0],         # other-vehicle
    5: [30, 30, 255],       # person
    6: [200, 40, 255],      # bicyclist
    7: [90, 30, 150],       # motorcyclist
    8: [255, 0, 255],       # road
    9: [255, 150, 255],     # parking
    10: [75, 0, 75],        # sidewalk
    11: [75, 0, 175],       # other-ground
    12: [0, 200, 255],      # building
    13: [50, 120, 255],     # fence
    14: [0, 175, 0],        # vegetation
    15: [0, 60, 135],       # trunk
    16: [80, 240, 150],     # terrain
    17: [150, 240, 255],    # pole
    18: [0, 0, 255],        # traffic-sign

    254: [0, 0, 0],    # invalid
    255: [255, 255, 255],        # unlabeled
}

learning_map = {
    0: 255,  # "unlabeled",
    1: 0,  # "car",
    2: 1,  # "bicycle",
    3: 2,  # "motorcycle",
    4: 3,  # "truck",
    5: 4,  # "other-vehicle",
    6: 5,  # "person",
    7: 6,  # "bicyclist",
    8: 7,  # "motorcyclist",
    9: 8,  # "road",
    10: 9,  # "parking",
    11: 10,  # "sidewalk",
    12: 11,  # "other-ground",
    13: 12,  # "building",
    14: 13,  # "fence",
    15: 14,  # "vegetation",
    16: 15,  # "trunk",
    17: 16,  # "terrain",
    18: 17,  # "pole",
    19: 18,  # "traffic-sign",
    20: 255  # "invalid"
}

learning_map_vis = learning_map.copy()
learning_map_vis[20] = 254   # invalid 单独可视化标签

learning_map_inv = {
    255: 0,
    0: 1, 1: 2, 2: 3, 3: 4,
    4: 5, 5: 6, 6: 7, 7: 8,
    8: 9, 9: 10, 10: 11, 11: 12,
    12: 13, 13: 14, 14: 15, 15: 16,
    16: 17, 17: 18
}


def get_color_for_label(label):
    label = int(label)
    if label in color_map:
        return color_map[label]
    else:
        print(f"Unknown label: {label}")
        return [0, 0, 0]

def save_to_txt(save_dir, filename, points, labels=None, prefix="original"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{prefix}_{filename}")

    with open(save_path, 'w') as f:
        f.write("x,y,z,intensity,label,label_color_R,label_color_G,label_color_B\n")

        for i in range(len(points)):
            x, y, z, intensity = points[i]
            label = int(labels[i]) if labels is not None else 255
            color = get_color_for_label(label)

            f.write(
                f"{x:.6f},{y:.6f},{z:.6f},{intensity:.6f},{label},{color[0]},{color[1]},{color[2]}\n"
            )

    print(f"已保存到: {save_path}")
    return save_path


def read_stf_and_save_txt(index, data_root, save_dir, prediction_label_dir=None):

    seq_files = [
        os.path.join(data_root, x) for x in os.listdir(data_root) if x.endswith('.bin')
    ]

    if index >= len(seq_files):
        print(f"索引 {index} 超出范围，共有 {len(seq_files)} 个文件")
        return

    # 可视化 LUT
    remap_dict = learning_map_vis
    max_key = max(remap_dict.keys())
    remap_lut = np.ones((max_key + 100), dtype=np.int32) * 255
    remap_lut[list(remap_dict.keys())] = list(remap_dict.values())
    label_map = remap_lut

    # 读取点云
    point_file = seq_files[index]
    with open(point_file, 'rb') as b:
        block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 5)[:, :4]
        block_[:, 3] /= 255.

    block = np.zeros_like(block_)
    block[:, :3] = block_[:, :3]
    block[:, 3] = block_[:, 3]

    # 读取原始标签
    label_file = point_file.replace('velodyne', 'labels').replace('.bin', '.label')
    labels_ = None

    if os.path.exists(label_file):
        with open(label_file, 'rb') as a:
            raw_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
            labels_ = label_map[raw_labels].astype(np.int64)

    base_filename = os.path.splitext(os.path.basename(point_file))[0]

    # 保存原始标签 TXT
    original_save_path = save_to_txt(
        save_dir,
        f"{base_filename}.txt",
        block,
        labels_,
        prefix="original"
    )

    # 预测标签
    if prediction_label_dir and os.path.exists(prediction_label_dir):

        pred_file = os.path.join(prediction_label_dir, f"{base_filename}.label")

        if os.path.exists(pred_file):
            with open(pred_file, 'rb') as f:
                pred_raw = np.fromfile(f, dtype=np.int32).reshape(-1)

            pred_labels = label_map[pred_raw].astype(np.int64)

            prediction_save_path = save_to_txt(
                save_dir,
                f"{base_filename}.txt",
                block,
                pred_labels,
                prefix="prediction"
            )

            print(f"已保存预测结果到: {prediction_save_path}")
        else:
            print(f"未找到预测标签文件: {pred_file}")

    return {
        "original_save_path": original_save_path,
        "point_count": len(block),
        "has_prediction": prediction_label_dir is not None
    }

if __name__ == "__main__":

    data_root = "/home/dataset/SemanticSTF/val/velodyne"
    save_dir = "/home/SemanticSTFV2/runs/pointdr_63456867/vis_label"
    prediction_label_dir = "/home/SemanticSTFV2/runs/pointdr_63456867/vis"

    seq_files = [f for f in os.listdir(data_root) if f.endswith('.bin')]
    file_count = len(seq_files)
    print(f"找到 {file_count} 个点云文件")

    for i in range(min(file_count, 251)):
        print(f"\n处理第 {i + 1}/{file_count} 个文件...")
        read_stf_and_save_txt(
            index=i,
            data_root=data_root,
            save_dir=save_dir,
            prediction_label_dir=prediction_label_dir
        )

    print("\n处理完成！")
