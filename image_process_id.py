#!/usr/bin/env python
# coding: utf-8


import requests
import numpy as np
import copy
import os
import random
import sys
import glob

from PIL import Image, ImageDraw
from skimage.util import random_noise
from ftplib import FTP
from io import BytesIO
from datetime import datetime

def get_image(url):
    return Image.open(requests.get(url, stream=True).raw)

def get_image_local(imgpath):
    return Image.open(imgpath)

# 裁剪 fix the border cal error 20201020
def cut(image, img_prefix, output_base_dir="./"):
    width, height = image.size
    # i代表裁剪后的相似度，9表示裁剪后的面积是原图的0.9=90%，那么边长应该是0.9^0.5=0.94868，因此border的比例应该是(1-0.9^0.5)/2
    for i in range(1, 10):
        border_ratio = (1-pow(i / 10, 0.5)) / 2
        #box = __box_to_int((width * fraction, height * fraction, width * (1 - fraction), height * (1 - fraction)))
        box = __box_to_int((width * border_ratio, height * border_ratio, width * (1 - border_ratio), height * (1 - border_ratio)))
        new_im = image.crop(box)
        #save file
        __save_file(new_im, "cut", img_prefix, f"{i/10:.1f}", output_base_dir=output_base_dir)

# 仿射变换
def geometrical_transform(image, img_prefix, output_base_dir="./"):

    # 旋转
    for i in range(0, 360, 45):
        new_im = image.rotate(i)
        # new_im.show()
        __save_file(new_im, "rotate", img_prefix, f"{i}deg", output_base_dir=output_base_dir)


    # 左右镜像
    new_im = image.transpose(Image.FLIP_LEFT_RIGHT)
    __save_file(new_im, "flip_left_right", img_prefix, "lr", output_base_dir=output_base_dir)

    # 上下镜像
    new_im = image.transpose(Image.FLIP_TOP_BOTTOM)
    __save_file(new_im, "flip_up_down", img_prefix, "ud", output_base_dir=output_base_dir)

# 加随机噪声，默认高斯
def add_noise(image, img_prefix, output_base_dir="./"):
    im_arr = np.asarray(image)

    # random_noise() method will convert image in [0, 255] to [0, 1.0],
    # inherently it use np.random.normal() to create normal distribution
    # and adds the generated noised back to image

    for i in range(1, 10):
        standard_deviation = i / 10  # 标准差

        # 默认高斯，其它可选mode: 'poisson', 'salt', 'pepper' ……
        noise_img = random_noise(im_arr, mode='gaussian', var=(1-standard_deviation) ** 2)
        noise_img = (255 * noise_img).astype(np.uint8)

        __save_file(Image.fromarray(noise_img), "noise", img_prefix, f"{standard_deviation:.1f}", output_base_dir=output_base_dir)


# 压缩图片
def compress_image(image, img_prefix, output_base_dir="./"):

    for i in range(1, 10):
        __save_file(image, "compress", img_prefix, f"q{i*10}", i * 10, output_base_dir=output_base_dir)


# 缩放图片
def resize(srcImg, img_prefix, output_base_dir="./"):
    w,h=srcImg.size
    # 得到一组不同resize值的图，用图片面积比例来代表相似度
    for i in range(10,100,10):
        j = pow(i / 100.0, 2)
        outImg=srcImg.resize((int(w*j),int(h*j)),Image.LANCZOS)
        __save_file(outImg, "resize", img_prefix, f"{i}pct", output_base_dir=output_base_dir)


# 拼接图片，高度匹配原图，待拼接图片会覆盖原有图片
def join_image_match_height(original_image, image2, img_prefix, output_base_dir="./"):
    original_width, height = (int(x) for x in original_image.size)
    original_image.resize((original_width, height))

    for i in range(1, 10):
        copyed_image2 = copy.deepcopy(image2)

        fraction = i / 10
        assert fraction < 1

        # 缩放带拼接图宽度
        width_of_image2 = int(copyed_image2.size[0] * height / copyed_image2.size[1])
        copyed_image2 = copyed_image2.resize((width_of_image2, height))

        # 超出原图的宽度（可能不超出）
        extra_width = int(fraction * original_width + width_of_image2 - original_width)
        extra_width = 0 if extra_width < 0 else extra_width

        output_image = Image.new(original_image.mode, (original_width + extra_width, height))
        output_image.paste(original_image, __box_to_int((0, 0)))
        output_image.paste(copyed_image2, __box_to_int((fraction * original_width, 0)))
        __save_file(output_image, "join", img_prefix, f"{fraction:.1f}", output_base_dir=output_base_dir)


# 添加马赛克效果
def add_mosaic(image, img_prefix, output_base_dir="./"):
    """
    添加马赛克效果，通过降低图像分辨率然后放大来实现
    """
    width, height = image.size
    
    for i in range(1, 10):
        # 马赛克块大小，i越大马赛克越明显
        mosaic_size = i * 2  # 马赛克块大小从2x2到18x18
        
        # 计算缩小后的尺寸
        small_width = max(1, width // mosaic_size)
        small_height = max(1, height // mosaic_size)
        
        # 先缩小再放大，产生马赛克效果
        small_img = image.resize((small_width, small_height), Image.NEAREST)
        mosaic_img = small_img.resize((width, height), Image.NEAREST)
        
        __save_file(mosaic_img, "mosaic", img_prefix, f"size{mosaic_size}", output_base_dir=output_base_dir)


# 添加干扰线
def add_interference_lines(image, img_prefix, output_base_dir="./"):
    """
    在图像上添加随机干扰线条
    """
    width, height = image.size
    
    for i in range(1, 10):
        # 创建图像副本
        img_with_lines = image.copy()
        draw = ImageDraw.Draw(img_with_lines)
        
        # 线条数量，i越大线条越多
        num_lines = i * 5  # 从5条到45条线
        
        for _ in range(num_lines):
            # 随机生成线条的起点和终点
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(0, width)
            y2 = random.randint(0, height)
            
            # 随机颜色（RGB）
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            
            # 随机线宽
            line_width = random.randint(1, 3)
            
            # 绘制线条
            draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
        
        __save_file(img_with_lines, "interference_lines", img_prefix, f"{num_lines}lines", output_base_dir=output_base_dir)


# 添加网格干扰
def add_grid_interference(image, img_prefix, output_base_dir="./"):
    """
    在图像上添加网格干扰
    """
    width, height = image.size
    
    for i in range(1, 10):
        # 创建图像副本
        img_with_grid = image.copy()
        draw = ImageDraw.Draw(img_with_grid)
        
        # 网格间距，i越大网格越密
        grid_spacing = max(10, 100 - i * 8)  # 从92像素间距到18像素间距
        
        # 网格颜色（半透明效果）
        grid_color = (128, 128, 128)  # 灰色
        line_width = 1
        
        # 绘制垂直线
        for x in range(0, width, grid_spacing):
            draw.line([(x, 0), (x, height)], fill=grid_color, width=line_width)
        
        # 绘制水平线
        for y in range(0, height, grid_spacing):
            draw.line([(0, y), (width, y)], fill=grid_color, width=line_width)
        
        __save_file(img_with_grid, "grid", img_prefix, f"spacing{grid_spacing}", output_base_dir=output_base_dir)

def gen_alltypes_imgs_by_url(original_image_url, image2_url, img_prefix, image2_prefix, output_base_dir="./transformed_images"):
    """
    通过URL生成所有类型的变换图片
    Args:
        original_image_url: 原始图片URL
        image2_url: 第二张图片URL（用于拼接）
        img_prefix: 图片前缀名（也作为子文件夹名）
        image2_prefix: 第二张图片前缀名
        output_base_dir: 输出基础目录，默认为"./transformed_images"
    """
    reset_counter()  # 重置计数器
    original_iamge = get_image(original_image_url)
    compress_image(original_iamge, img_prefix, output_base_dir)
    add_noise(original_iamge, img_prefix, output_base_dir)
    geometrical_transform(original_iamge, img_prefix, output_base_dir)
    cut(original_iamge, img_prefix, output_base_dir)
    resize(original_iamge, img_prefix, output_base_dir)
    add_mosaic(original_iamge, img_prefix, output_base_dir)
    add_interference_lines(original_iamge, img_prefix, output_base_dir)
    add_grid_interference(original_iamge, img_prefix, output_base_dir)
    image2 = get_image(image2_url)
    join_image_match_height(original_iamge, image2, img_prefix+"_"+image2_prefix, output_base_dir)

def gen_alltypes_imgs_by_local_path(original_image_path, image2_path, img_prefix, image2_prefix, output_base_dir="./transformed_images"):
    """
    通过本地路径生成所有类型的变换图片
    Args:
        original_image_path: 原始图片本地路径
        image2_path: 第二张图片本地路径（用于拼接）
        img_prefix: 图片前缀名（也作为子文件夹名）
        image2_prefix: 第二张图片前缀名
        output_base_dir: 输出基础目录，默认为"./transformed_images"
    """
    reset_counter()  # 重置计数器
    original_iamge = get_image_local(original_image_path)
    compress_image(original_iamge, img_prefix, output_base_dir)
    add_noise(original_iamge, img_prefix, output_base_dir)
    geometrical_transform(original_iamge, img_prefix, output_base_dir)
    cut(original_iamge, img_prefix, output_base_dir)
    resize(original_iamge, img_prefix, output_base_dir)
    add_mosaic(original_iamge, img_prefix, output_base_dir)
    add_interference_lines(original_iamge, img_prefix, output_base_dir)
    add_grid_interference(original_iamge, img_prefix, output_base_dir)
    image2 = get_image_local(image2_path)
    join_image_match_height(original_iamge, image2, img_prefix+"_"+image2_prefix, output_base_dir)  

# 全局计数器，用于生成连续编号
_global_counter = 0

def _get_next_number():
    """获取下一个图片编号"""
    global _global_counter
    _global_counter += 1
    return _global_counter

def reset_counter():
    """重置计数器"""
    global _global_counter
    _global_counter = 0

# 保存图片到本地
def __save_file(image, transform_type, img_prefix, param_value, quality=100, output_base_dir="./"):
    """
    保存图片到指定路径
    Args:
        image: 要保存的图片
        transform_type: 变换类型名称
        img_prefix: 图片前缀名
        param_value: 参数值
        quality: 图片质量
        output_base_dir: 输出基础目录路径
    """
    # 创建图片专属文件夹路径：output_base_dir/img_prefix/
    img_dir = os.path.join(output_base_dir, img_prefix)
    
    # 创建目录（递归创建）
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    image_tmp = image.convert('RGB')
    
    # 获取下一个编号
    img_number = _get_next_number()
    
    # 新的命名格式：编号_图片名_变化方式_参数.jpg
    filename = f"{img_number}_{img_prefix}_{transform_type}_{param_value}.jpg"
    filepath = os.path.join(img_dir, filename)
    image_tmp.save(filepath, quality=quality)



def __box_to_int(box):
    return tuple(int(i) for i in box)


# ==================== 交互式功能 ====================

def interactive_path_selection():
    """
    交互式路径选择功能
    返回用户选择的输入路径、输出路径和处理模式
    """
    print("\n🎯 图像变换工具 - 交互式配置")
    print("=" * 50)
    
    # 选择处理模式
    print("\n📋 请选择处理模式:")
    print("1. 处理单张图片")
    print("2. 批量处理文件夹中的图片")
    print("3. 处理两张图片（包含拼接功能）")
    print("4. 通过URL处理图片")
    
    while True:
        try:
            mode = int(input("\n请输入选择 (1-4): "))
            if mode in [1, 2, 3, 4]:
                break
            else:
                print("❌ 请输入有效数字 (1-4)")
        except ValueError:
            print("❌ 请输入有效数字")
    
    return mode


def get_input_path(mode):
    """
    根据模式获取输入路径
    """
    if mode == 1:  # 单张图片
        print("\n📁 请指定输入图片路径:")
        return get_file_path("图片文件")
    
    elif mode == 2:  # 批量处理
        print("\n📁 请指定输入文件夹路径:")
        return get_directory_path("输入文件夹")
    
    elif mode == 3:  # 两张图片
        print("\n📁 请指定第一张图片路径:")
        path1 = get_file_path("第一张图片")
        print("\n📁 请指定第二张图片路径:")
        path2 = get_file_path("第二张图片")
        return path1, path2
    
    elif mode == 4:  # URL模式
        print("\n🌐 请输入第一张图片的URL:")
        url1 = input("URL1: ").strip()
        print("\n🌐 请输入第二张图片的URL (用于拼接):")
        url2 = input("URL2: ").strip()
        return url1, url2


def get_file_path(file_type):
    """
    获取文件路径，支持相对路径和绝对路径
    """
    while True:
        print(f"\n💡 输入提示:")
        print(f"- 可以输入相对路径: ./demo_images/image.jpg")
        print(f"- 可以输入绝对路径: /Users/username/Pictures/image.jpg")
        print(f"- 输入 'browse' 浏览当前目录下的图片文件")
        
        path = input(f"\n请输入{file_type}路径: ").strip()
        
        if path.lower() == 'browse':
            selected_file = browse_files()
            if selected_file:
                return selected_file
            continue
        
        if os.path.isfile(path):
            return os.path.abspath(path)
        else:
            print(f"❌ 文件不存在: {path}")
            retry = input("是否重新输入? (y/n): ").strip().lower()
            if retry != 'y':
                sys.exit("用户取消操作")


def get_directory_path(dir_type):
    """
    获取目录路径
    """
    while True:
        print(f"\n💡 输入提示:")
        print(f"- 可以输入相对路径: ./demo_images")
        print(f"- 可以输入绝对路径: /Users/username/Pictures")
        print(f"- 输入 'browse' 浏览当前目录")
        
        path = input(f"\n请输入{dir_type}路径: ").strip()
        
        if path.lower() == 'browse':
            selected_dir = browse_directories()
            if selected_dir:
                return selected_dir
            continue
            
        if os.path.isdir(path):
            return os.path.abspath(path)
        else:
            print(f"❌ 目录不存在: {path}")
            retry = input("是否重新输入? (y/n): ").strip().lower()
            if retry != 'y':
                sys.exit("用户取消操作")


def browse_files():
    """
    浏览并选择文件
    """
    current_dir = os.getcwd()
    print(f"\n📂 当前目录: {current_dir}")
    
    # 查找图片文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(ext))
        image_files.extend(glob.glob(ext.upper()))
    
    if not image_files:
        print("❌ 当前目录下没有找到图片文件")
        return None
    
    print(f"\n📷 找到 {len(image_files)} 个图片文件:")
    for i, file in enumerate(image_files, 1):
        print(f"{i:2d}. {file}")
    
    while True:
        try:
            choice = int(input(f"\n请选择文件 (1-{len(image_files)}): "))
            if 1 <= choice <= len(image_files):
                selected = os.path.abspath(image_files[choice - 1])
                print(f"✅ 已选择: {selected}")
                return selected
            else:
                print(f"❌ 请输入 1-{len(image_files)} 之间的数字")
        except ValueError:
            print("❌ 请输入有效数字")


def browse_directories():
    """
    浏览并选择目录
    """
    current_dir = os.getcwd()
    print(f"\n📂 当前目录: {current_dir}")
    
    # 获取所有子目录
    dirs = [d for d in os.listdir('.') if os.path.isdir(d) and not d.startswith('.')]
    
    if not dirs:
        print("❌ 当前目录下没有子目录")
        return None
    
    print(f"\n📁 找到 {len(dirs)} 个子目录:")
    dirs.append(".")  # 添加当前目录选项
    
    for i, dir_name in enumerate(dirs, 1):
        if dir_name == ".":
            print(f"{i:2d}. {dir_name} (当前目录)")
        else:
            print(f"{i:2d}. {dir_name}")
    
    while True:
        try:
            choice = int(input(f"\n请选择目录 (1-{len(dirs)}): "))
            if 1 <= choice <= len(dirs):
                selected = os.path.abspath(dirs[choice - 1])
                print(f"✅ 已选择: {selected}")
                return selected
            else:
                print(f"❌ 请输入 1-{len(dirs)} 之间的数字")
        except ValueError:
            print("❌ 请输入有效数字")


def get_output_path():
    """
    获取输出路径
    """
    print(f"\n📁 请指定输出目录:")
    print(f"💡 输入提示:")
    print(f"- 直接回车使用默认路径: ./transformed_images")
    print(f"- 可以输入相对路径: ./my_results")
    print(f"- 可以输入绝对路径: /Users/username/Documents/results")
    
    output_path = input("\n请输入输出目录路径 (回车使用默认): ").strip()
    
    if not output_path:
        output_path = "./transformed_images"
    
    output_path = os.path.abspath(output_path)
    
    # 如果目录不存在，询问是否创建
    if not os.path.exists(output_path):
        create = input(f"目录 {output_path} 不存在，是否创建? (y/n): ").strip().lower()
        if create == 'y':
            try:
                os.makedirs(output_path, exist_ok=True)
                print(f"✅ 已创建目录: {output_path}")
            except Exception as e:
                print(f"❌ 创建目录失败: {e}")
                sys.exit("无法创建输出目录")
        else:
            sys.exit("用户取消操作")
    
    return output_path


def get_custom_prefix():
    """
    获取自定义前缀名
    """
    print(f"\n🏷️  图片前缀名设置:")
    print(f"💡 前缀名将用作文件名和子文件夹名")
    print(f"- 直接回车使用默认名称（从文件名自动生成）")
    print(f"- 或输入自定义前缀名")
    
    prefix = input("\n请输入图片前缀名 (回车使用默认): ").strip()
    return prefix if prefix else None


def show_processing_summary(mode, input_path, output_path, prefix=None):
    """
    显示处理摘要
    """
    print(f"\n📋 处理配置摘要:")
    print("=" * 40)
    
    mode_names = {
        1: "单张图片处理",
        2: "批量文件夹处理", 
        3: "两张图片处理（含拼接）",
        4: "URL图片处理"
    }
    
    print(f"🎯 处理模式: {mode_names[mode]}")
    
    if mode in [1, 2]:
        print(f"📁 输入路径: {input_path}")
    elif mode == 3:
        print(f"📁 输入路径1: {input_path[0]}")
        print(f"📁 输入路径2: {input_path[1]}")
    elif mode == 4:
        print(f"🌐 输入URL1: {input_path[0]}")
        print(f"🌐 输入URL2: {input_path[1]}")
    
    print(f"📂 输出路径: {output_path}")
    
    if prefix:
        print(f"🏷️  图片前缀: {prefix}")
    else:
        print(f"🏷️  图片前缀: 自动生成")
    
    print("=" * 40)
    
    confirm = input("\n确认开始处理? (y/n): ").strip().lower()
    return confirm == 'y'


def interactive_main():
    """
    交互式主函数
    """
    try:
        # 1. 选择处理模式
        mode = interactive_path_selection()
        
        # 2. 获取输入路径
        input_path = get_input_path(mode)
        
        # 3. 获取输出路径
        output_path = get_output_path()
        
        # 4. 获取自定义前缀（可选）
        custom_prefix = get_custom_prefix()
        
        # 5. 显示摘要并确认
        if not show_processing_summary(mode, input_path, output_path, custom_prefix):
            print("❌ 用户取消操作")
            return
        
        # 6. 执行处理
        print(f"\n🚀 开始处理...")
        
        if mode == 1:  # 单张图片
            gen_single_image_transforms(
                image_path=input_path,
                img_prefix=custom_prefix,
                output_base_dir=output_path
            )
            
        elif mode == 2:  # 批量处理
            batch_transform_images(
                image_dir=input_path,
                output_base_dir=output_path
            )
            
        elif mode == 3:  # 两张图片
            path1, path2 = input_path
            prefix1 = custom_prefix or os.path.splitext(os.path.basename(path1))[0]
            prefix2 = input("请输入第二张图片的前缀名: ").strip() or os.path.splitext(os.path.basename(path2))[0]
            
            gen_alltypes_imgs_by_local_path(
                original_image_path=path1,
                image2_path=path2,
                img_prefix=prefix1,
                image2_prefix=prefix2,
                output_base_dir=output_path
            )
            
        elif mode == 4:  # URL处理
            url1, url2 = input_path
            prefix1 = custom_prefix or "web_image1"
            prefix2 = input("请输入第二张图片的前缀名: ").strip() or "web_image2"
            
            gen_alltypes_imgs_by_url(
                original_image_url=url1,
                image2_url=url2,
                img_prefix=prefix1,
                image2_prefix=prefix2,
                output_base_dir=output_path
            )
        
        print(f"\n🎉 处理完成！")
        print(f"📁 结果保存在: {output_path}")
        
    except KeyboardInterrupt:
        print(f"\n\n❌ 用户中断操作")
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


# 便捷函数：只对单张图片进行变换（不需要第二张图片）
def gen_single_image_transforms(image_path, img_prefix=None, output_base_dir="./transformed_images"):
    """
    对单张图片进行所有变换（除了拼接）
    Args:
        image_path: 图片路径
        img_prefix: 图片前缀名，如果不提供则从文件名自动生成
        output_base_dir: 输出基础目录
    """
    if img_prefix is None:
        # 从文件路径自动生成前缀名
        img_prefix = os.path.splitext(os.path.basename(image_path))[0]
    
    reset_counter()  # 重置计数器
    original_image = get_image_local(image_path)
    compress_image(original_image, img_prefix, output_base_dir)
    add_noise(original_image, img_prefix, output_base_dir)
    geometrical_transform(original_image, img_prefix, output_base_dir)
    cut(original_image, img_prefix, output_base_dir)
    resize(original_image, img_prefix, output_base_dir)
    add_mosaic(original_image, img_prefix, output_base_dir)
    add_interference_lines(original_image, img_prefix, output_base_dir)
    add_grid_interference(original_image, img_prefix, output_base_dir)
    
    print(f"✅ 已完成图片 '{img_prefix}' 的所有变换，保存到: {output_base_dir}/{img_prefix}/")
    return output_base_dir


# 批量处理多张图片
def batch_transform_images(image_dir, output_base_dir="./transformed_images", file_extensions=None):
    """
    批量处理文件夹中的所有图片
    Args:
        image_dir: 输入图片文件夹路径
        output_base_dir: 输出基础目录
        file_extensions: 支持的文件扩展名列表，默认为常见图片格式
    """
    if file_extensions is None:
        file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    image_dir = os.path.abspath(image_dir)
    if not os.path.exists(image_dir):
        print(f"❌ 输入目录不存在: {image_dir}")
        return
    
    # 获取所有图片文件
    image_files = []
    for ext in file_extensions:
        pattern = os.path.join(image_dir, f"*{ext}")
        pattern_upper = os.path.join(image_dir, f"*{ext.upper()}")
        import glob
        image_files.extend(glob.glob(pattern))
        image_files.extend(glob.glob(pattern_upper))
    
    if not image_files:
        print(f"⚠️  在目录 {image_dir} 中未找到支持的图片文件")
        return
    
    print(f"🔍 找到 {len(image_files)} 张图片，开始批量处理...")
    
    success_count = 0
    for i, image_path in enumerate(image_files, 1):
        try:
            img_name = os.path.splitext(os.path.basename(image_path))[0]
            print(f"[{i}/{len(image_files)}] 处理图片: {img_name}")
            gen_single_image_transforms(image_path, img_name, output_base_dir)
            success_count += 1
        except Exception as e:
            print(f"❌ 处理图片 {image_path} 失败: {e}")
    
    print(f"\n📊 批量处理完成: {success_count}/{len(image_files)} 张图片处理成功")
    print(f"📁 结果保存在: {os.path.abspath(output_base_dir)}")
    return output_base_dir

# def __output_file(file_name, image):
#     temp = BytesIO()
#     image.save(temp, format="png")
#     FTP.storbinary('STOR ' + file_name, temp)


if __name__ == "__main__":
    print("🔧 图像变换工具")
    print("=" * 50)
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive" or sys.argv[1] == "-i":
            # 交互式模式
            interactive_main()
        elif sys.argv[1] == "--demo":
            # 演示模式
            print("📚 演示模式 - 自动处理demo_images文件夹")
            demo_dir = "demo_images"
            if os.path.exists(demo_dir):
                demo_files = glob.glob(os.path.join(demo_dir, "*.jpg")) + glob.glob(os.path.join(demo_dir, "*.png"))
                if demo_files:
                    print(f"📷 示例1: 处理单张图片")
                    sample_image = demo_files[0]
                    print(f"   输入图片: {sample_image}")
                    gen_single_image_transforms(sample_image, output_base_dir="./example_output")
                    print()
                
                print(f"📚 示例2: 批量处理demo_images文件夹")
                batch_transform_images(demo_dir, output_base_dir="./batch_output")
            else:
                print("❌ demo_images目录不存在")
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            # 帮助信息
            print("📖 使用方法:")
            print("  python image_process_id.py --interactive  # 交互式模式")
            print("  python image_process_id.py --demo         # 演示模式")
            print("  python image_process_id.py --help         # 显示帮助")
            print("  python image_process_id.py               # 默认交互式模式")
            print("\n💡 功能说明:")
            print("1. 使用 gen_single_image_transforms(图片路径, 输出目录) 处理单张图片")
            print("2. 使用 batch_transform_images(文件夹路径, 输出目录) 批量处理")
            print("3. 使用 gen_alltypes_imgs_by_local_path() 处理需要拼接的两张图片")
            print("4. 所有变换结果会按图片名分组到各自的子文件夹中")
        else:
            print(f"❌ 未知参数: {sys.argv[1]}")
            print("使用 --help 查看帮助信息")
    else:
        # 默认进入交互式模式
        print("💡 默认进入交互式模式 (使用 --help 查看更多选项)")
        print()
        interactive_main()
    
    # 原始示例（如果文件存在的话）
    # gen_alltypes_imgs_by_local_path("1fa9d61f2b00b0c752cdf017e7f1c65aed54567c.png",\
    #     "J9KeYkEZf4HHD5LRGf799N-1024-80.jpg",\
    #     "1fa9d61f2b00b0c752cdf017e7f1c65aed54567c",\
    #     "J9KeYkEZf4HHD5LRGf799N")
