#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量图像增强工具启动脚本
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

def check_dependencies():
    """检查依赖项"""
    missing_deps = []
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
        
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
        
    try:
        from PIL import Image, ImageTk
    except ImportError:
        missing_deps.append("Pillow")
        
    try:
        # 检查imgaug库
        sys.path.append(os.path.join(os.path.dirname(__file__), 'pkg'))
        import imgaug
        import imgaug.augmenters as iaa
    except ImportError as e:
        missing_deps.append(f"imgaug (错误: {str(e)})")
        
    return missing_deps

def install_dependencies():
    """安装依赖项"""
    import subprocess
    import sys
    
    deps = [
        "numpy",
        "opencv-python",
        "Pillow",
        "scikit-image",
        "scipy"
    ]
    
    print("正在安装依赖项...")
    for dep in deps:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✓ {dep} 安装成功")
        except subprocess.CalledProcessError:
            print(f"✗ {dep} 安装失败")
            return False
    return True

def main():
    """主函数"""
    import time
    start_time = time.time()
    print("批量图像增强工具启动中...")
    
    # 检查依赖项
    deps_start = time.time()
    missing_deps = check_dependencies()
    deps_time = time.time() - deps_start
    print(f"依赖项检查耗时: {deps_time:.2f}秒")
    
    if missing_deps:
        print(f"缺少依赖项: {', '.join(missing_deps)}")
        
        # 创建简单的GUI询问是否安装
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        result = messagebox.askyesno(
            "缺少依赖项",
            f"检测到缺少以下依赖项:\n{', '.join(missing_deps)}\n\n是否自动安装这些依赖项？"
        )
        
        if result:
            if install_dependencies():
                messagebox.showinfo("成功", "依赖项安装完成，程序将重新启动")
                # 重新启动程序
                os.execv(sys.executable, ['python'] + sys.argv)
            else:
                messagebox.showerror("错误", "依赖项安装失败，请手动安装")
                return
        else:
            messagebox.showinfo("提示", "请手动安装依赖项后重新运行程序")
            return
    
    # 启动主程序
    try:
        ui_start = time.time()
        from batch_image_augmentation_advanced import AdvancedBatchImageAugmentation
        import_time = time.time() - ui_start
        print(f"模块导入耗时: {import_time:.2f}秒")
        
        init_start = time.time()
        root = tk.Tk()
        app = AdvancedBatchImageAugmentation(root)
        init_time = time.time() - init_start
        print(f"界面初始化耗时: {init_time:.2f}秒")
        
        total_time = time.time() - start_time
        print(f"总启动时间: {total_time:.2f}秒")
        
        root.mainloop()
        
    except Exception as e:
        print(f"程序启动失败: {e}")
        messagebox.showerror("错误", f"程序启动失败:\n{str(e)}")

if __name__ == "__main__":
    main() 