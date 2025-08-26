#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量图像增强工具 - 高级版本
基于imgaug库的图形界面应用，支持配置文件保存和加载
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
import traceback

# 添加imgaug库路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'pkg'))

# 延迟导入imgaug库以提高启动速度
ia = None
iaa = None

def _lazy_import_imgaug():
    """延迟导入imgaug库"""
    global ia, iaa
    if ia is None:
        import imgaug as _ia
        import imgaug.augmenters as _iaa
        ia = _ia
        iaa = _iaa
    return ia, iaa

class AdvancedBatchImageAugmentation:
    def __init__(self, root):
        self.root = root
        self.root.title("批量图像增强工具 v2.0 - 高级版")
        
        # 设置自适应窗口
        self.setup_responsive_window()
        
        # 变量初始化
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.augmentation_count = tk.IntVar(value=5)
        self.progress_var = tk.DoubleVar()
        self.is_processing = False
        
        # 配置管理
        self.config_file = "config.json"
        self.config = self.load_config()
        self.augmenters_config = {}
        
        # 设置主题颜色
        self.set_theme()
    
    def setup_responsive_window(self):
        """设置自适应窗口"""
        # 获取屏幕尺寸
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # 设置窗口为屏幕的85%大小，确保有足够空间显示所有内容
        window_width = int(screen_width * 0.85)
        window_height = int(screen_height * 0.85)
        
        # 确保最小尺寸能显示完整界面
        min_width = max(1200, window_width)
        min_height = max(800, window_height)
        
        # 计算居中位置
        x = (screen_width - min_width) // 2
        y = (screen_height - min_height) // 2
        
        # 设置窗口大小和位置
        self.root.geometry(f"{min_width}x{min_height}+{x}+{y}")
        
        # 设置最小窗口大小，确保增强选择区域可见
        self.root.minsize(1200, 800)
        
        # 允许窗口调整大小
        self.root.resizable(True, True)
        
        # 绑定窗口大小改变事件
        self.root.bind('<Configure>', self.on_window_resize)
    
    def on_window_resize(self, event):
        """窗口大小改变时的回调"""
        # 只处理主窗口的resize事件
        if event.widget == self.root:
            # 这里可以添加响应式布局调整逻辑
            pass
    
    def set_theme(self):
        """设置应用主题"""
        style = ttk.Style()
        
        # 配置主题颜色
        style.configure("TFrame", background="#f5f5f5")
        style.configure("TLabel", background="#f5f5f5")
        style.configure("TLabelframe", background="#f5f5f5")
        style.configure("TLabelframe.Label", font=("Arial", 10, "bold"))
        style.configure("TButton", font=("Arial", 9))
        style.configure("TCheckbutton", background="#f5f5f5")
        
        # 设置选项卡样式
        style.configure("TNotebook.Tab", padding=[12, 4], font=("Arial", 10))
        
        # 为增强器选项添加特殊样式
        style.configure("Large.TCheckbutton", 
                       font=("Arial", 12, "bold"), 
                       background="#f8f8f8")
        
        style.configure("Card.TFrame", 
                       background="#ffffff", 
                       relief="raised", 
                       borderwidth=1)
        
        # 类别标题样式
        style.configure("CategoryHeader.TFrame",
                       background="#ecf0f1",
                       relief="solid",
                       borderwidth=1)
        
        # 选项卡片样式
        style.configure("OptionCard.TFrame",
                       background="#ffffff",
                       relief="solid",
                       borderwidth=1)
        
        # Mac风格滚动条样式
        style.configure("Mac.Vertical.TScrollbar",
                       background="#e1e1e1",
                       troughcolor="#f0f0f0",
                       borderwidth=0,
                       arrowcolor="#999999",
                       darkcolor="#e1e1e1",
                       lightcolor="#e1e1e1")
        
        # 设置UI
        self.setup_ui()
        self.load_last_settings()
        
    def load_config(self):
        """加载配置文件 - 优化版本"""
        default_config = {
            "default_settings": {
                "augmentation_count": 5,
                "random_seed": 42,
                "keep_size": True,
                "last_input_folder": "",
                "last_output_folder": ""
            },
            "augmenter_categories": {}
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 合并默认配置，确保所有必要的键都存在
                    for key in default_config:
                        if key not in config:
                            config[key] = default_config[key]
                    return config
            else:
                return default_config
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return default_config
            
    def save_config(self):
        """保存配置文件"""
        try:
            # 保存当前设置
            self.config["default_settings"]["augmentation_count"] = self.augmentation_count.get()
            self.config["default_settings"]["random_seed"] = self.seed_var.get()
            self.config["default_settings"]["keep_size"] = self.keep_size_var.get()
            self.config["default_settings"]["last_input_folder"] = self.input_folder.get()
            self.config["default_settings"]["last_output_folder"] = self.output_folder.get()
            
            # 保存增强器状态
            for aug_name, config in self.augmenters_config.items():
                if aug_name in self.config.get("augmenter_categories", {}):
                    self.config["augmenter_categories"][aug_name]["enabled"] = config["var"].get()
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            print(f"保存配置文件失败: {e}")
            
    def load_last_settings(self):
        """加载上次的设置"""
        if "default_settings" in self.config:
            settings = self.config["default_settings"]
            self.augmentation_count.set(settings.get("augmentation_count", 5))
            self.input_folder.set(settings.get("last_input_folder", ""))
            self.output_folder.set(settings.get("last_output_folder", ""))
        
    def setup_ui(self):
        """设置用户界面"""
        # 创建菜单栏
        self.create_menu()
        
        # 主框架
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重 - 自适应布局
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)  # 增强器选择区域可扩展
        
        # 创建各个区域
        self.create_file_selection_frame(main_frame)
        self.create_augmenter_selection_frame(main_frame)
        self.create_parameter_frame(main_frame)
        self.create_control_frame(main_frame)
        self.create_log_frame(main_frame)
        
    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="保存配置", command=self.save_config)
        file_menu.add_command(label="加载配置", command=self.load_config_from_file)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说明", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)
        
    def create_file_selection_frame(self, parent):
        """创建文件选择框架"""
        file_frame = ttk.LabelFrame(parent, text="文件选择", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 输入文件夹选择
        ttk.Label(file_frame, text="输入文件夹:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.input_folder, width=60).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="浏览", command=self.select_input_folder).grid(row=0, column=2, padx=5, pady=5)
        
        # 输出文件夹选择
        ttk.Label(file_frame, text="输出文件夹:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.output_folder, width=60).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="浏览", command=self.select_output_folder).grid(row=1, column=2, padx=5, pady=5)
        
        # 增强数量设置
        ttk.Label(file_frame, text="每张图片增强数量:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(file_frame, from_=1, to=50, textvariable=self.augmentation_count, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 快速选择按钮
        ttk.Button(file_frame, text="选择测试图片", command=self.select_test_images).grid(row=2, column=2, padx=5, pady=5)
        
    def create_augmenter_selection_frame(self, parent):
        """创建增强器选择框架"""
        augmenter_frame = ttk.LabelFrame(parent, text="增强方式选择", padding="10")
        augmenter_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        augmenter_frame.columnconfigure(0, weight=1)
        augmenter_frame.rowconfigure(0, weight=1)
        
        # 创建带滚动条的框架 - Mac风格
        canvas = tk.Canvas(augmenter_frame, highlightthickness=0, background="#f8f8f8", height=400)
        # Mac风格的滚动条
        scrollbar = ttk.Scrollbar(augmenter_frame, orient="vertical", command=canvas.yview, style="Mac.Vertical.TScrollbar")
        scrollable_frame = ttk.Frame(canvas)
        
        # Mac优化的滚动功能
        def _on_mousewheel(event):
            # 支持Mac触摸板的精确滚动
            try:
                if event.delta:
                    # Mac触摸板通常产生较小的delta值，提供更平滑的滚动
                    if abs(event.delta) < 5:
                        # 触摸板双指滑动 - 更精细的滚动
                        canvas.yview_scroll(int(-1*event.delta), "units")
                    else:
                        # 鼠标滚轮 - 标准滚动
                        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                else:
                    # 备用处理
                    canvas.yview_scroll(-1 if event.num == 4 else 1, "units")
            except:
                # 兜底处理
                canvas.yview_scroll(-1, "units")
        
        def _on_shift_mousewheel(event):
            # 支持Shift+滚轮的水平滚动（如果需要）
            canvas.xview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            # 绑定滚动事件到当前widget
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel)
            # Mac 触摸板双指滑动支持
            canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
            canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Shift-MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")
        
        # 绑定鼠标进入和离开事件
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        # 直接绑定触摸板事件到canvas
        canvas.bind("<MouseWheel>", _on_mousewheel)
        canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))
        
        # 配置滚动区域
        def _configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        scrollable_frame.bind("<Configure>", _configure_scroll_region)
        
        # 创建canvas窗口
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 当canvas大小改变时，调整scrollable_frame宽度
        def _configure_canvas(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", _configure_canvas)
        
        # 定义增强器类别和选项
        self.augmenter_categories = {
            "几何变换": {
                "Affine": {"scale": (0.8, 1.2), "rotate": (-15, 15), "translate_percent": (-0.1, 0.1)},
                "Rotate": {"rotate": (-30, 30)},
                "Resize": {"size": (0.7, 1.3)},
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
        
        # 重新设计增强器选择界面 - 清晰的布局
        current_row = 0
        
        for category, augmenters in self.augmenter_categories.items():
            # 类别标题框架
            category_header = ttk.Frame(scrollable_frame, style="CategoryHeader.TFrame")
            category_header.grid(row=current_row, column=0, columnspan=3, sticky=(tk.W, tk.E), 
                               pady=(15, 5), padx=5)
            category_header.columnconfigure(0, weight=1)
            
            # 类别标题
            title_label = ttk.Label(category_header, text=f"【{category}】", 
                                  font=("Arial", 12, "bold"), foreground="#2c3e50")
            title_label.pack(side=tk.LEFT)
            
            # 全选按钮
            select_all_var = tk.BooleanVar()
            select_all_btn = ttk.Button(category_header, text="全选", width=6,
                                      command=lambda cat=category, var=select_all_var: self.select_category_all(cat, var))
            select_all_btn.pack(side=tk.RIGHT, padx=(5, 0))
            
            current_row += 1
            
            # 增强器选项 - 3列网格布局
            aug_list = list(augmenters.items())
            for i, (aug_name, params) in enumerate(aug_list):
                col = i % 3
                if col == 0 and i > 0:
                    current_row += 1
                
                # 创建选项框架
                option_frame = ttk.Frame(scrollable_frame, style="OptionCard.TFrame")
                option_frame.grid(row=current_row, column=col, sticky=(tk.W, tk.E, tk.N), 
                                padx=8, pady=5)
                
                var = tk.BooleanVar()
                
                # 复选框
                cb = ttk.Checkbutton(option_frame, text=aug_name, variable=var)
                cb.pack(anchor=tk.W, pady=(8, 2), padx=8)
                
                # 描述文字
                desc = self.get_augmenter_description(aug_name)
                if desc:
                    desc_label = ttk.Label(option_frame, text=desc, font=("Arial", 9), 
                                         foreground="#666666", wraplength=200, justify="left")
                    desc_label.pack(anchor=tk.W, pady=(0, 8), padx=(8, 8))
                
                # 存储配置
                self.augmenters_config[aug_name] = {
                    "var": var,
                    "params": params,
                    "category": category
                }
            
            # 如果最后一行不满3个，需要移到下一行
            if len(aug_list) % 3 != 0:
                current_row += 1
            
            current_row += 1  # 类别间距
        
        # 配置scrollable_frame的列权重，让3列平均分配空间
        for i in range(3):
            scrollable_frame.columnconfigure(i, weight=1)
        
        # 配置滚动 - 带滚动条布局
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 设置列权重
        augmenter_frame.columnconfigure(0, weight=1)
        augmenter_frame.columnconfigure(1, weight=0)  # 滚动条列不扩展
        augmenter_frame.rowconfigure(0, weight=1)
        
        # 添加操作提示
        hint_frame = ttk.Frame(augmenter_frame)
        hint_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(hint_frame, text="💡 提示：", font=("Arial", 10, "bold"), foreground="#2E8B57").pack(side=tk.LEFT)
        ttk.Label(hint_frame, text="支持触摸板双指滑动或鼠标滚轮滚动", font=("Arial", 10), foreground="#555555").pack(side=tk.LEFT, padx=(8, 0))
        
        # 添加快速选择按钮
        quick_frame = ttk.Frame(hint_frame)
        quick_frame.pack(side=tk.RIGHT)
        
        ttk.Button(quick_frame, text="全部选中", width=8, 
                  command=self.select_all_augmenters).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_frame, text="全部取消", width=8, 
                  command=self.deselect_all_augmenters).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_frame, text="随机推荐", width=8, 
                  command=self.select_random_recommended_augmenters).pack(side=tk.LEFT, padx=2)
        
    def get_augmenter_description(self, aug_name):
        """获取增强器描述"""
        descriptions = {
            "Affine": "仿射变换：缩放、旋转、平移",
            "Rotate": "旋转：-30到30度",
            "Scale": "缩放：70%到130%",
            "Translate": "平移：-20%到20%",
            "Shear": "剪切：-15到15度",
            "PerspectiveTransform": "透视变换",
            "ElasticTransformation": "弹性变换",
            "AddToBrightness": "亮度调整：-30到30",
            "MultiplyBrightness": "亮度乘法：0.7到1.3倍",
            "AddToHue": "色调调整：-20到20",
            "AddToSaturation": "饱和度调整：-30到30",
            "Grayscale": "灰度化：50%到100%",
            "ChangeColorTemperature": "色温调整：1000K到11000K",
            "Posterize": "色调分离：3到7位",
            "GaussianBlur": "高斯模糊：0到1.0",
            "AverageBlur": "平均模糊：2到7",
            "MedianBlur": "中值模糊：3到7",
            "MotionBlur": "运动模糊：3到7，角度-45到45",
            "AdditiveGaussianNoise": "高斯噪声：0到12.75",
            "AdditivePoissonNoise": "泊松噪声：0到10",
            "SaltAndPepper": "椒盐噪声：0到5%",
            "ContrastNormalization": "对比度归一化：0.5到1.5",
            "HistogramEqualization": "直方图均衡化",
            "CLAHE": "CLAHE：限制1到4，网格3到7",
            "Sharpen": "锐化：强度0到1，亮度0.75到1.25",
            "Emboss": "浮雕：强度0到1，强度0.5到1.5",
            "Clouds": "云朵：密度0到30%",
            "Rain": "雨滴：长度和宽度0.1到0.3",
            "Snowflakes": "雪花：大小和密度0.1到0.3",
            "Fog": "雾：密度0到30%",
            "Canny": "Canny边缘检测：强度0到1",
            "DirectedEdgeDetect": "定向边缘检测：强度0到1",
            "FrequencyNoiseAlpha": "频率噪声：指数-4到4，最大尺寸4到16",
            "SimplexNoiseAlpha": "Simplex噪声：最大尺寸4到16"
        }
        return descriptions.get(aug_name, "")
        
    def select_category_all(self, category, var):
        """选择/取消选择整个类别"""
        for aug_name, config in self.augmenters_config.items():
            if config["category"] == category:
                config["var"].set(var.get())
                
    def select_all_augmenters(self):
        """选择所有增强器"""
        for aug_name, config in self.augmenters_config.items():
            config["var"].set(True)
            
    def deselect_all_augmenters(self):
        """取消选择所有增强器"""
        for aug_name, config in self.augmenters_config.items():
            config["var"].set(False)
            
    def select_random_recommended_augmenters(self):
        """随机选择推荐的增强器组合"""
        import random
        
        # 先全部取消
        self.deselect_all_augmenters()
        
        # 按类别定义的优质增强器池
        category_pools = {
            "几何变换": ["Affine", "Rotate", "Resize", "PerspectiveTransform"],
            "颜色增强": ["AddToBrightness", "MultiplyBrightness", "AddToHue", "AddToSaturation", "Grayscale"],
            "模糊和噪声": ["GaussianBlur", "AverageBlur", "AdditiveGaussianNoise", "SaltAndPepper"],
            "对比度和锐化": ["ContrastNormalization", "CLAHE", "Sharpen", "Emboss"],
            "天气效果": ["Clouds", "Rain", "Snowflakes"],
            "边缘和纹理": ["Canny", "DirectedEdgeDetect"]
        }
        
        selected_augmenters = []
        
        # 从每个类别随机选择1-2个增强器
        for category, augmenters in category_pools.items():
            # 随机选择该类别要选几个增强器（1-2个）
            num_to_select = random.randint(1, min(2, len(augmenters)))
            # 随机选择增强器
            selected_from_category = random.sample(augmenters, num_to_select)
            selected_augmenters.extend(selected_from_category)
        
        # 确保总数不超过10个（避免选择过多）
        if len(selected_augmenters) > 10:
            selected_augmenters = random.sample(selected_augmenters, 10)
        
        # 应用选择
        for aug_name, config in self.augmenters_config.items():
            if aug_name in selected_augmenters:
                config["var"].set(True)
        
        # 显示选择了多少个增强器
        from tkinter import messagebox
        messagebox.showinfo("随机推荐", f"已随机选择 {len(selected_augmenters)} 个增强器组合！")
                
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
        
        # 输出格式选择
        self.output_format = tk.StringVar(value="png")
        ttk.Label(param_frame, text="输出格式:").grid(row=3, column=0, sticky=tk.W, pady=2)
        format_combo = ttk.Combobox(param_frame, textvariable=self.output_format, values=["png", "jpg", "bmp", "tiff"], width=10)
        format_combo.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
    def create_control_frame(self, parent):
        """创建控制按钮框架"""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 按钮框架
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        # 开始处理按钮
        self.process_btn = ttk.Button(button_frame, text="开始批量增强", command=self.start_processing)
        self.process_btn.grid(row=0, column=0, padx=5, pady=5)
        
        # 停止处理按钮
        self.stop_btn = ttk.Button(button_frame, text="停止处理", command=self.stop_processing, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # 预览按钮
        self.preview_btn = ttk.Button(button_frame, text="预览效果", command=self.preview_augmentation)
        self.preview_btn.grid(row=0, column=2, padx=5, pady=5)
        
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
        
        # 清空日志按钮
        ttk.Button(log_frame, text="清空日志", command=self.clear_log).grid(row=1, column=0, pady=5)
        
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
            
    def select_test_images(self):
        """选择测试图片文件夹"""
        # 使用项目中的测试图片
        test_folder = os.path.join(os.path.dirname(__file__), "data", "img")
        if os.path.exists(test_folder):
            self.input_folder.set(test_folder)
            self.log_message(f"选择测试图片文件夹: {test_folder}")
        else:
            messagebox.showinfo("提示", "未找到测试图片文件夹，请手动选择输入文件夹")
            
    def log_message(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)
        
    def get_selected_augmenters(self):
        """获取选中的增强器"""
        selected = []
        for aug_name, config in self.augmenters_config.items():
            if config["var"].get():
                selected.append((aug_name, config))
        return selected
        
    def create_augmenter_pipeline(self, selected_augmenters):
        """创建增强器管道"""
        # 延迟导入imgaug
        ia, iaa = _lazy_import_imgaug()
        
        augmenters = []
        
        for aug_name, config in selected_augmenters:
            try:
                if aug_name == "Affine":
                    aug = iaa.Affine(**config["params"])
                elif aug_name == "Rotate":
                    aug = iaa.Rotate(**config["params"])
                elif aug_name == "Resize":
                    aug = iaa.Resize(**config["params"])
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
                    aug = iaa.CLAHE(clip_limit=config["params"].get("clip_limit", (1, 4)))
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
        
    def preview_augmentation(self):
        """预览增强效果"""
        if not self.input_folder.get():
            messagebox.showerror("错误", "请先选择输入文件夹")
            return
            
        selected_augmenters = self.get_selected_augmenters()
        if not selected_augmenters:
            messagebox.showerror("错误", "请至少选择一种增强方式")
            return
            
        # 创建预览窗口
        self.create_preview_window(selected_augmenters)
        
    def create_preview_window(self, selected_augmenters):
        """创建预览窗口"""
        preview_window = tk.Toplevel(self.root)
        preview_window.title("增强效果预览")
        preview_window.geometry("800x600")
        
        # 获取第一张图片进行预览
        input_path = Path(self.input_folder.get())
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in input_path.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            messagebox.showerror("错误", "输入文件夹中没有找到图像文件")
            return
            
        # 读取第一张图片
        image_file = image_files[0]
        image = cv2.imread(str(image_file))
        if image is None:
            messagebox.showerror("错误", f"无法读取图像 {image_file.name}")
            return
            
        # 创建增强器管道
        augmenters = self.create_augmenter_pipeline(selected_augmenters)
        if not augmenters:
            messagebox.showerror("错误", "没有有效的增强器")
            return
            
        # 延迟导入imgaug
        ia, iaa = _lazy_import_imgaug()
        
        pipeline = iaa.Sequential(augmenters, random_order=True)
        ia.seed(self.seed_var.get())
        
        # 应用增强
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented_image = pipeline.augment_image(image_rgb)
        augmented_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
        
        # 显示原图和增强后的图片
        self.show_image_comparison(preview_window, image, augmented_bgr, image_file.name)
        
    def show_image_comparison(self, window, original, augmented, filename):
        """显示图像对比"""
        # 创建框架
        frame = ttk.Frame(window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        ttk.Label(frame, text=f"预览: {filename}", font=("Arial", 12, "bold")).pack(pady=10)
        
        # 图像显示框架
        image_frame = ttk.Frame(frame)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # 原图
        original_frame = ttk.LabelFrame(image_frame, text="原图")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # 增强图
        augmented_frame = ttk.LabelFrame(image_frame, text="增强后")
        augmented_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # 调整图像大小用于显示
        def resize_for_display(img, max_size=300):
            height, width = img.shape[:2]
            if height > max_size or width > max_size:
                scale = min(max_size / height, max_size / width)
                new_height = int(height * scale)
                new_width = int(width * scale)
                return cv2.resize(img, (new_width, new_height))
            return img
        
        # 显示图像
        original_resized = resize_for_display(original)
        augmented_resized = resize_for_display(augmented)
        
        # 转换为PIL图像用于显示
        original_pil = Image.fromarray(cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB))
        augmented_pil = Image.fromarray(cv2.cvtColor(augmented_resized, cv2.COLOR_BGR2RGB))
        
        original_photo = ImageTk.PhotoImage(original_pil)
        augmented_photo = ImageTk.PhotoImage(augmented_pil)
        
        # 创建标签显示图像
        original_label = ttk.Label(original_frame, image=original_photo)
        original_label.image = original_photo  # 保持引用
        original_label.pack(pady=10)
        
        augmented_label = ttk.Label(augmented_frame, image=augmented_photo)
        augmented_label.image = augmented_photo  # 保持引用
        augmented_label.pack(pady=10)
        
        # 关闭按钮
        ttk.Button(frame, text="关闭", command=window.destroy).pack(pady=10)
        
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
            
        # 保存配置
        self.save_config()
        
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
                
            # 延迟导入imgaug
            ia, iaa = _lazy_import_imgaug()
            
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
                    original_output = output_subdir / f"{image_file.stem}_original.{self.output_format.get()}"
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
                            output_file = output_subdir / f"{image_file.stem}_aug_{j+1:02d}.{self.output_format.get()}"
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
            
    def load_config_from_file(self):
        """从文件加载配置"""
        filename = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self.load_last_settings()
                self.log_message(f"成功加载配置文件: {filename}")
                messagebox.showinfo("成功", "配置文件加载成功")
            except Exception as e:
                messagebox.showerror("错误", f"加载配置文件失败:\n{str(e)}")
                
    def show_help(self):
        """显示帮助信息"""
        help_text = """
使用说明：

1. 文件选择：
   - 选择包含图像的输入文件夹
   - 选择输出文件夹用于保存增强后的图像
   - 设置每张图片的增强数量

2. 增强方式选择：
   - 勾选需要使用的增强方式
   - 每个类别都有"全选"按钮
   - 可以组合多种增强方式

3. 参数设置：
   - 设置随机种子确保结果可重现
   - 选择是否保持原始尺寸
   - 选择输出图像格式

4. 预览功能：
   - 点击"预览效果"查看增强效果
   - 可以调整参数后重新预览

5. 批量处理：
   - 点击"开始批量增强"开始处理
   - 可以随时停止处理
   - 查看处理日志了解进度

6. 配置管理：
   - 程序会自动保存配置
   - 可以通过菜单加载其他配置文件

支持的图像格式：JPG, PNG, BMP, TIFF
        """
        messagebox.showinfo("使用说明", help_text)
        
    def show_about(self):
        """显示关于信息"""
        about_text = """
批量图像增强工具 v2.0

基于imgaug库开发的图形界面应用
支持多种图像增强方式

功能特点：
- 支持30+种增强方式
- 图形化界面操作
- 批量处理功能
- 实时预览效果
- 配置保存和加载
- 详细的处理日志

作者：基于imgaug项目开发
        """
        messagebox.showinfo("关于", about_text)

def main():
    """主函数"""
    root = tk.Tk()
    app = AdvancedBatchImageAugmentation(root)
    root.mainloop()

if __name__ == "__main__":
    main() 