import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

LOG_FILE_NAME = '/home/SemanticSTFV2/runs/aug_base_robust_62573289/summary/scalars.jsonl'
OUTPUT_IMAGE_NAME = '/home/SemanticSTFV2/runs/aug_base_robust_62573289/loss_curve.png'


def plot_and_save_all_loss_curve():
    all_step_data = {}
    loss_names = set()

    print(f"--- 启动稀疏数据绘图程序 ---")
    print(f"正在尝试读取文件: {LOG_FILE_NAME}...")

    try:
        with open(LOG_FILE_NAME, 'r') as f:
            line_count = 0

            for line in f:
                line_count += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    if 'global_step' not in data:
                        continue

                    current_step = int(data['global_step'])

                    # 记录该 step 的数据
                    step_losses = {}
                    for key, value in data.items():
                        if key.startswith('loss'):
                            try:
                                loss_value = float(value)
                                step_losses[key] = loss_value
                                loss_names.add(key)  # 记录所有发现的损失名称
                            except (ValueError, TypeError):
                                pass

                    # 将数据存入主字典
                    if step_losses:
                        all_step_data[current_step] = step_losses

                except json.JSONDecodeError:
                    print(f"[错误] 第 {line_count} 行 JSON 格式错误，跳过。")
                    continue

        # 1. 结果校验
        if not all_step_data:
            print("\n[失败] 未读取到任何有效的 global_step 或 loss 数据。请检查文件内容。")
            sys.exit(1)

        print(f"\n[成功] 共读取 {len(all_step_data)} 个有效 global_step 数据点。")
        print(f"将绘制的损失曲线包括: {sorted(list(loss_names))}")

        # 2. 准备绘图数据 (对齐数据)
        # 获取所有 step 的有序列表作为 X 轴
        global_steps_sorted = sorted(all_step_data.keys())

        # 准备 Y 轴的损失数据字典 {loss_name: [value1, value2, ...], ...}
        plot_data = {name: [] for name in loss_names}

        for step in global_steps_sorted:
            step_record = all_step_data[step]

            for name in loss_names:
                # 如果该 step 记录了某个 loss，则使用该值；否则使用 np.nan 填充
                value = step_record.get(name, np.nan)
                plot_data[name].append(value)

        # --- 3. 绘制图表 ---
        plt.figure(figsize=(14, 8))

        for loss_name, losses in plot_data.items():
            # 格式化标签名
            label_name = loss_name.replace('_', ' ').title().replace('Loss ', '')

            # 绘制曲线。Matplotlib 会自动跳过 np.nan 的点。
            plt.plot(global_steps_sorted, losses,
                     label=label_name,
                     linewidth=1.5,
                     alpha=0.8)

            # 4. 设置图表样式
        plt.title('Training Loss Curves (Sparse Data Handled)', fontsize=18, fontweight='bold')
        plt.xlabel('Global Step', fontsize=14)
        plt.ylabel('Loss Value', fontsize=14)

        plt.legend(loc='best', fontsize=12, title="Loss Components")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # --- 5. 保存图片 ---
        plt.savefig(OUTPUT_IMAGE_NAME, dpi=300)
        print(f"\n[完成] 损失曲线图已成功保存为图片: {os.path.abspath(OUTPUT_IMAGE_NAME)}")

    except FileNotFoundError:
        print(f"\n[致命错误] 找不到文件 '{LOG_FILE_NAME}'。请检查文件名和路径是否正确。")
    except Exception as e:
        print(f"\n[致命错误] 发生未知错误: {e}")

if __name__ == "__main__":
    plot_and_save_all_loss_curve()