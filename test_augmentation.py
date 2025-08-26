#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像增强功能测试脚本
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 添加imgaug库路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'pkg'))
import imgaug as ia
import imgaug.augmenters as iaa

def test_basic_augmentation():
    """测试基本增强功能"""
    print("开始测试基本增强功能...")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 测试各种增强器
    augmenters = [
        ("旋转", iaa.Rotate(rotate=(-15, 15))),
        ("缩放", iaa.Resize(size=(0.8, 1.2))),
        ("亮度调整", iaa.AddToBrightness(add=(-30, 30))),
        ("高斯模糊", iaa.GaussianBlur(sigma=(0.0, 1.0))),
        ("高斯噪声", iaa.AdditiveGaussianNoise(loc=0, scale=(0, 0.05*255))),
        ("对比度调整", iaa.ContrastNormalization(alpha=(0.5, 1.5))),
    ]
    
    for name, aug in augmenters:
        try:
            # 应用增强
            augmented = aug.augment_image(test_image)
            print(f"✓ {name} 测试通过")
        except Exception as e:
            print(f"✗ {name} 测试失败: {e}")
            
def test_pipeline_augmentation():
    """测试增强管道"""
    print("\n开始测试增强管道...")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 创建增强管道
    pipeline = iaa.Sequential([
        iaa.Rotate(rotate=(-15, 15)),
        iaa.Resize(size=(0.8, 1.2)),
        iaa.AddToBrightness(add=(-30, 30)),
        iaa.GaussianBlur(sigma=(0.0, 1.0)),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0, 0.05*255)),
        iaa.ContrastNormalization(alpha=(0.5, 1.5))
    ], random_order=True)
    
    try:
        # 应用管道增强
        augmented = pipeline.augment_image(test_image)
        print("✓ 增强管道测试通过")
    except Exception as e:
        print(f"✗ 增强管道测试失败: {e}")
        
def test_image_loading():
    """测试图像加载功能"""
    print("\n开始测试图像加载功能...")
    
    # 检查测试图像文件夹
    test_folder = Path("data/img")
    if test_folder.exists():
        image_files = list(test_folder.glob("*.png"))
        if image_files:
            try:
                # 加载第一张图像
                image_path = str(image_files[0])
                image = cv2.imread(image_path)
                if image is not None:
                    print(f"✓ 图像加载测试通过: {image_files[0].name}")
                    print(f"  图像尺寸: {image.shape}")
                else:
                    print(f"✗ 图像加载失败: {image_files[0].name}")
            except Exception as e:
                print(f"✗ 图像加载测试失败: {e}")
        else:
            print("⚠ 未找到测试图像文件")
    else:
        print("⚠ 测试图像文件夹不存在")
        
def test_augmenter_creation():
    """测试增强器创建"""
    print("\n开始测试增强器创建...")
    
    # 测试所有支持的增强器
    augmenter_tests = [
        ("Affine", lambda: iaa.Affine(scale=(0.8, 1.2), rotate=(-15, 15))),
        ("Rotate", lambda: iaa.Rotate(rotate=(-30, 30))),
        ("Resize", lambda: iaa.Resize(size=(0.7, 1.3))),
        ("Translate", lambda: iaa.Translate(translate_percent=(-0.2, 0.2))),
        ("Shear", lambda: iaa.Shear(shear=(-15, 15))),
        ("AddToBrightness", lambda: iaa.AddToBrightness(add=(-30, 30))),
        ("MultiplyBrightness", lambda: iaa.MultiplyBrightness(mul=(0.7, 1.3))),
        ("AddToHue", lambda: iaa.AddToHue(value=(-20, 20))),
        ("AddToSaturation", lambda: iaa.AddToSaturation(value=(-30, 30))),
        ("Grayscale", lambda: iaa.Grayscale(alpha=(0.5, 1.0))),
        ("GaussianBlur", lambda: iaa.GaussianBlur(sigma=(0.0, 1.0))),
        ("AverageBlur", lambda: iaa.AverageBlur(k=(2, 7))),
        ("MedianBlur", lambda: iaa.MedianBlur(k=(3, 7))),
        ("MotionBlur", lambda: iaa.MotionBlur(k=(3, 7), angle=(-45, 45))),
        ("AdditiveGaussianNoise", lambda: iaa.AdditiveGaussianNoise(loc=0, scale=(0, 0.05*255))),
        ("AdditivePoissonNoise", lambda: iaa.AdditivePoissonNoise(lam=(0, 10))),
        ("SaltAndPepper", lambda: iaa.SaltAndPepper(p=(0, 0.05))),
        ("ContrastNormalization", lambda: iaa.ContrastNormalization(alpha=(0.5, 1.5))),
        ("HistogramEqualization", lambda: iaa.HistogramEqualization()),
        ("CLAHE", lambda: iaa.CLAHE(clip_limit=(1, 4), tile_grid_size=(3, 7))),
        ("Sharpen", lambda: iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 1.25))),
        ("Emboss", lambda: iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))),
    ]
    
    success_count = 0
    total_count = len(augmenter_tests)
    
    for name, creator in augmenter_tests:
        try:
            aug = creator()
            success_count += 1
            print(f"✓ {name} 创建成功")
        except Exception as e:
            print(f"✗ {name} 创建失败: {e}")
            
    print(f"\n增强器创建测试结果: {success_count}/{total_count} 成功")
    
def test_save_and_load():
    """测试保存和加载功能"""
    print("\n开始测试保存和加载功能...")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 测试保存
    try:
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        # 保存原始图像
        cv2.imwrite(str(output_dir / "test_original.png"), test_image)
        
        # 创建增强器并保存增强图像
        aug = iaa.Rotate(rotate=(-15, 15))
        augmented = aug.augment_image(test_image)
        cv2.imwrite(str(output_dir / "test_augmented.png"), augmented)
        
        print("✓ 图像保存测试通过")
        
        # 清理测试文件
        import shutil
        shutil.rmtree(output_dir)
        
    except Exception as e:
        print(f"✗ 图像保存测试失败: {e}")
        
def main():
    """主测试函数"""
    print("=" * 50)
    print("图像增强功能测试")
    print("=" * 50)
    
    # 运行所有测试
    test_basic_augmentation()
    test_pipeline_augmentation()
    test_image_loading()
    test_augmenter_creation()
    test_save_and_load()
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)
    
    # 检查imgaug版本
    try:
        print(f"imgaug版本: {ia.__version__}")
    except:
        print("无法获取imgaug版本信息")
        
    # 检查OpenCV版本
    print(f"OpenCV版本: {cv2.__version__}")
    
    # 检查NumPy版本
    print(f"NumPy版本: {np.__version__}")

if __name__ == "__main__":
    main() 