#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量图像增强工具
基于imgaug库的图形界面应用
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import threading
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageTk
import json
from datetime import datetime

# 添加imgaug库路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'pkg'))
import imgaug as ia
import imgaug.augmenters as iaa

class BatchImageAugmentation:
    def __init__(self, root):
        self.root = root
        self.root.title("批量图像增强工具 v1.0")
        self.root.geometry("1200x800")
        
        # 变量初始化
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.augmentation_count = tk.IntVar(value=5)
        self.progress_var = tk.DoubleVar()
        self.is_processing = False
        
        # 增强器配置
        self.augmenters_config = {}
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 文件选择区域
        self.create_file_selection_frame(main_frame)
        
        # 增强器选择区域
        self.create_augmenter_selection_frame(main_frame)
        
        # 参数设置区域
        self.create_parameter_frame(main_frame)
        
        # 控制按钮区域
        self.create_control_frame(main_frame)
        
        # 日志显示区域
        self.create_log_frame(main_frame)
        
    def create_file_selection_frame(self, parent):
        """创建文件选择框架"""
        file_frame = ttk.LabelFrame(parent, text="文件选择", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 输入文件夹选择
        ttk.Label(file_frame, text="输入文件夹:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.input_folder, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="浏览", command=self.select_input_folder).grid(row=0, column=2, padx=5, pady=5)
        
        # 输出文件夹选择
        ttk.Label(file_frame, text="输出文件夹:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.output_folder, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="浏览", command=self.select_output_folder).grid(row=1, column=2, padx=5, pady=5)
        
        # 增强数量设置
        ttk.Label(file_frame, text="每张图片增强数量:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(file_frame, from_=1, to=20, textvariable=self.augmentation_count, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
    def create_augmenter_selection_frame(self, parent):
        """创建增强器选择框架"""
        augmenter_frame = ttk.LabelFrame(parent, text="增强方式选择", padding="10")
        augmenter_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        augmenter_frame.columnconfigure(0, weight=1)
        augmenter_frame.rowconfigure(0, weight=1)
        
        # 创建滚动框架
        canvas = tk.Canvas(augmenter_frame)
        scrollbar = ttk.Scrollbar(augmenter_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 定义增强器类别和选项
        self.augmenter_categories = {
            "几何变换": {
                "Affine": {"scale": (0.8, 1.2), "rotate": (-15, 15), "translate_percent": (-0.1, 0.1)},
                "Rotate": {"rotate": (-30, 30)},
                "Scale": {"scale": (0.7, 1.3)},
                "Translate": {"translate_percent": (-0.2, 0.2)},
                "Shear": {"shear": (-15, 15)},
                "PerspectiveTransform": {"scale": (0.01, 0.1)},
                "ElasticTransformation": {"alpha": (0, 50), "sigma": (4, 8)}
            },
            "颜色增强": {
                "AddToBrightness": {"add": (-30, 30)},
                "MultiplyBrightness": {"mul": (0.7, 1.3)},
                "AddToHue": {"value": (-20, 20)},
                "AddToSaturation": {"value": (-30, 30)},
                "Grayscale": {"alpha": (0.5, 1.0)},
                "ChangeColorTemperature": {"kelvin": (1000, 11000)},
                "Posterize": {"nb_bits": (3, 7)}
            },
            "模糊和噪声": {
                "GaussianBlur": {"sigma": (0.0, 1.0)},
                "AverageBlur": {"k": (2, 7)},
                "MedianBlur": {"k": (3, 7)},
                "MotionBlur": {"k": (3, 7), "angle": (-45, 45)},
                "AdditiveGaussianNoise": {"loc": 0, "scale": (0, 0.05*255)},
                "AdditivePoissonNoise": {"lam": (0, 10)},
                "SaltAndPepper": {"p": (0, 0.05)}
            },
            "对比度和锐化": {
                "ContrastNormalization": {"alpha": (0.5, 1.5)},
                "HistogramEqualization": {},
                "CLAHE": {"clip_limit": (1, 4), "tile_grid_size": (3, 7)},
                "Sharpen": {"alpha": (0.0, 1.0), "lightness": (0.75, 1.25)},
                "Emboss": {"alpha": (0.0, 1.0), "strength": (0.5, 1.5)}
            },
            "天气效果": {
                "Clouds": {"density": (0.0, 0.3)},
                "Rain": {"drop_length": (0.1, 0.3), "drop_width": (0.1, 0.3)},
                "Snowflakes": {"flake_size": (0.1, 0.3), "flake_density": (0.1, 0.3)},
                "Fog": {"density": (0.0, 0.3)}
            },
            "边缘和纹理": {
                "Canny": {"alpha": (0.0, 1.0)},
                "DirectedEdgeDetect": {"alpha": (0.0, 1.0)},
                "FrequencyNoiseAlpha": {"exponent": (-4, 4), "size_px_max": (4, 16)},
                "SimplexNoiseAlpha": {"size_px_max": (4, 16)}
            }
        }
        
        # 创建增强器选择界面
        row = 0
        for category, augmenters in self.augmenter_categories.items():
            # 类别标题
            ttk.Label(scrollable_frame, text=f"【{category}】", font=("Arial", 10, "bold")).grid(
                row=row, column=0, columnspan=3, sticky=tk.W, pady=(10, 5)
            )
            row += 1
            
            # 增强器选项
            for i, (aug_name, params) in enumerate(augmenters.items()):
                col = i % 3
                if col == 0 and i > 0:
                    row += 1
                
                # 创建复选框
                var = tk.BooleanVar()
                cb = ttk.Checkbutton(scrollable_frame, text=aug_name, variable=var)
                cb.grid(row=row, column=col, sticky=tk.W, padx=10, pady=2)
                
                # 存储配置
                self.augmenters_config[aug_name] = {
                    "var": var,
                    "params": params,
                    "category": category
                }
        
        # 配置滚动
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        augmenter_frame.columnconfigure(0, weight=1)
        augmenter_frame.rowconfigure(0, weight=1)
        
    def create_parameter_frame(self, parent):
        """创建参数设置框架"""
        param_frame = ttk.LabelFrame(parent, text="参数设置", padding="10")
        param_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 全局参数
        ttk.Label(param_frame, text="全局参数:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # 随机种子
        self.seed_var = tk.IntVar(value=42)
        ttk.Label(param_frame, text="随机种子:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(param_frame, textvariable=self.seed_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 保持尺寸
        self.keep_size_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="保持原始尺寸", variable=self.keep_size_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)
        
    def create_control_frame(self, parent):
        """创建控制按钮框架"""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 开始处理按钮
        self.process_btn = ttk.Button(control_frame, text="开始批量增强", command=self.start_processing)
        self.process_btn.grid(row=0, column=0, padx=5, pady=5)
        
        # 停止处理按钮
        self.stop_btn = ttk.Button(control_frame, text="停止处理", command=self.stop_processing, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # 进度条
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # 状态标签
        self.status_label = ttk.Label(control_frame, text="就绪")
        self.status_label.grid(row=2, column=0, columnspan=2, pady=5)
        
    def create_log_frame(self, parent):
        """创建日志显示框架"""
        log_frame = ttk.LabelFrame(parent, text="处理日志", padding="10")
        log_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # 日志文本框
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=100)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def select_input_folder(self):
        """选择输入文件夹"""
        folder = filedialog.askdirectory(title="选择输入文件夹")
        if folder:
            self.input_folder.set(folder)
            self.log_message(f"选择输入文件夹: {folder}")
            
    def select_output_folder(self):
        """选择输出文件夹"""
        folder = filedialog.askdirectory(title="选择输出文件夹")
        if folder:
            self.output_folder.set(folder)
            self.log_message(f"选择输出文件夹: {folder}")
            
    def log_message(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def get_selected_augmenters(self):
        """获取选中的增强器"""
        selected = []
        for aug_name, config in self.augmenters_config.items():
            if config["var"].get():
                selected.append((aug_name, config))
        return selected
        
    def create_augmenter_pipeline(self, selected_augmenters):
        """创建增强器管道"""
        augmenters = []
        
        for aug_name, config in selected_augmenters:
            try:
                if aug_name == "Affine":
                    aug = iaa.Affine(**config["params"])
                elif aug_name == "Rotate":
                    aug = iaa.Rotate(**config["params"])
                elif aug_name == "Scale":
                    aug = iaa.Scale(**config["params"])
                elif aug_name == "Translate":
                    aug = iaa.Translate(**config["params"])
                elif aug_name == "Shear":
                    aug = iaa.Shear(**config["params"])
                elif aug_name == "PerspectiveTransform":
                    aug = iaa.PerspectiveTransform(**config["params"])
                elif aug_name == "ElasticTransformation":
                    aug = iaa.ElasticTransformation(**config["params"])
                elif aug_name == "AddToBrightness":
                    aug = iaa.AddToBrightness(**config["params"])
                elif aug_name == "MultiplyBrightness":
                    aug = iaa.MultiplyBrightness(**config["params"])
                elif aug_name == "AddToHue":
                    aug = iaa.AddToHue(**config["params"])
                elif aug_name == "AddToSaturation":
                    aug = iaa.AddToSaturation(**config["params"])
                elif aug_name == "Grayscale":
                    aug = iaa.Grayscale(**config["params"])
                elif aug_name == "ChangeColorTemperature":
                    aug = iaa.ChangeColorTemperature(**config["params"])
                elif aug_name == "Posterize":
                    aug = iaa.Posterize(**config["params"])
                elif aug_name == "GaussianBlur":
                    aug = iaa.GaussianBlur(**config["params"])
                elif aug_name == "AverageBlur":
                    aug = iaa.AverageBlur(**config["params"])
                elif aug_name == "MedianBlur":
                    aug = iaa.MedianBlur(**config["params"])
                elif aug_name == "MotionBlur":
                    aug = iaa.MotionBlur(**config["params"])
                elif aug_name == "AdditiveGaussianNoise":
                    aug = iaa.AdditiveGaussianNoise(**config["params"])
                elif aug_name == "AdditivePoissonNoise":
                    aug = iaa.AdditivePoissonNoise(**config["params"])
                elif aug_name == "SaltAndPepper":
                    aug = iaa.SaltAndPepper(**config["params"])
                elif aug_name == "ContrastNormalization":
                    aug = iaa.ContrastNormalization(**config["params"])
                elif aug_name == "HistogramEqualization":
                    aug = iaa.HistogramEqualization(**config["params"])
                elif aug_name == "CLAHE":
                    aug = iaa.CLAHE(**config["params"])
                elif aug_name == "Sharpen":
                    aug = iaa.Sharpen(**config["params"])
                elif aug_name == "Emboss":
                    aug = iaa.Emboss(**config["params"])
                elif aug_name == "Clouds":
                    aug = iaa.Clouds(**config["params"])
                elif aug_name == "Rain":
                    aug = iaa.Rain(**config["params"])
                elif aug_name == "Snowflakes":
                    aug = iaa.Snowflakes(**config["params"])
                elif aug_name == "Fog":
                    aug = iaa.Fog(**config["params"])
                elif aug_name == "Canny":
                    aug = iaa.Canny(**config["params"])
                elif aug_name == "DirectedEdgeDetect":
                    aug = iaa.DirectedEdgeDetect(**config["params"])
                elif aug_name == "FrequencyNoiseAlpha":
                    aug = iaa.FrequencyNoiseAlpha(**config["params"])
                elif aug_name == "SimplexNoiseAlpha":
                    aug = iaa.SimplexNoiseAlpha(**config["params"])
                else:
                    self.log_message(f"警告: 未知的增强器 {aug_name}")
                    continue
                    
                augmenters.append(aug)
                self.log_message(f"添加增强器: {aug_name}")
                
            except Exception as e:
                self.log_message(f"错误: 创建增强器 {aug_name} 失败: {str(e)}")
                
        return augmenters
        
    def start_processing(self):
        """开始批量处理"""
        if not self.input_folder.get():
            messagebox.showerror("错误", "请选择输入文件夹")
            return
            
        if not self.output_folder.get():
            messagebox.showerror("错误", "请选择输出文件夹")
            return
            
        selected_augmenters = self.get_selected_augmenters()
        if not selected_augmenters:
            messagebox.showerror("错误", "请至少选择一种增强方式")
            return
            
        # 在新线程中处理
        self.is_processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        thread = threading.Thread(target=self.process_images, args=(selected_augmenters,))
        thread.daemon = True
        thread.start()
        
    def stop_processing(self):
        """停止处理"""
        self.is_processing = False
        self.process_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log_message("处理已停止")
        
    def process_images(self, selected_augmenters):
        """处理图像"""
        try:
            # 创建增强器管道
            augmenters = self.create_augmenter_pipeline(selected_augmenters)
            if not augmenters:
                self.log_message("错误: 没有有效的增强器")
                return
                
            # 组合增强器
            pipeline = iaa.Sequential(augmenters, random_order=True)
            
            # 设置随机种子
            ia.seed(self.seed_var.get())
            
            # 获取图像文件列表
            input_path = Path(self.input_folder.get())
            output_path = Path(self.output_folder.get())
            output_path.mkdir(exist_ok=True)
            
            # 支持的图像格式
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_files = [f for f in input_path.iterdir() 
                          if f.is_file() and f.suffix.lower() in image_extensions]
            
            if not image_files:
                self.log_message("错误: 输入文件夹中没有找到图像文件")
                return
                
            self.log_message(f"找到 {len(image_files)} 个图像文件")
            
            total_operations = len(image_files) * self.augmentation_count.get()
            processed_operations = 0
            
            # 处理每个图像文件
            for i, image_file in enumerate(image_files):
                if not self.is_processing:
                    break
                    
                try:
                    # 读取图像
                    image = cv2.imread(str(image_file))
                    if image is None:
                        self.log_message(f"警告: 无法读取图像 {image_file.name}")
                        continue
                        
                    # 转换为RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # 创建输出文件夹
                    output_subdir = output_path / image_file.stem
                    output_subdir.mkdir(exist_ok=True)
                    
                    # 保存原始图像
                    original_output = output_subdir / f"{image_file.stem}_original{image_file.suffix}"
                    cv2.imwrite(str(original_output), image)
                    
                    # 生成增强图像
                    for j in range(self.augmentation_count.get()):
                        if not self.is_processing:
                            break
                            
                        try:
                            # 应用增强
                            augmented_image = pipeline.augment_image(image_rgb)
                            
                            # 转换回BGR并保存
                            augmented_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                            output_file = output_subdir / f"{image_file.stem}_aug_{j+1:02d}{image_file.suffix}"
                            cv2.imwrite(str(output_file), augmented_bgr)
                            
                            processed_operations += 1
                            progress = (processed_operations / total_operations) * 100
                            self.progress_var.set(progress)
                            
                            # 更新状态
                            self.status_label.config(text=f"处理中: {image_file.name} ({j+1}/{self.augmentation_count.get()})")
                            
                        except Exception as e:
                            self.log_message(f"错误: 增强图像 {image_file.name} 第 {j+1} 次失败: {str(e)}")
                            
                except Exception as e:
                    self.log_message(f"错误: 处理图像 {image_file.name} 失败: {str(e)}")
                    
            if self.is_processing:
                self.log_message(f"处理完成! 共处理 {len(image_files)} 个文件，生成 {processed_operations} 个增强图像")
                self.status_label.config(text="处理完成")
                messagebox.showinfo("完成", f"批量增强完成!\n共处理 {len(image_files)} 个文件\n生成 {processed_operations} 个增强图像")
            else:
                self.log_message("处理已停止")
                
        except Exception as e:
            self.log_message(f"处理过程中发生错误: {str(e)}")
            messagebox.showerror("错误", f"处理过程中发生错误:\n{str(e)}")
            
        finally:
            self.is_processing = False
            self.process_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.progress_var.set(0)

def main():
    """主函数"""
    root = tk.Tk()
    app = BatchImageAugmentation(root)
    root.mainloop()

if __name__ == "__main__":
    main() 