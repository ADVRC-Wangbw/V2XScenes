#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量创建V2XScenes数据集软链接的Python脚本
Python script for batch creating symbolic links for V2XScenes dataset

使用方法 / Usage:
===============

1. 创建所有软链接 / Create all symbolic links:
   python create_v2x_links.py

2. 强制覆盖已存在的软链接 / Force overwrite existing symbolic links:
   python create_v2x_links.py --force

3. 跳过已存在的软链接 / Skip existing symbolic links:
   python create_v2x_links.py --skip-existing

4. 只创建指定的单个片段 / Create only a specific segment:
   python create_v2x_links.py --single "20240705_200812_300_1720181335_to_1720181350_15_39"

5. 列出所有可用片段 / List all available segments:
   python create_v2x_links.py --list

6. 自定义源和目标路径 / Custom source and target paths:
   python create_v2x_links.py --source /自定义/源路径 --target /自定义/目标路径
   python create_v2x_links.py --source /custom/source/path --target /custom/target/path

7. 组合使用 / Combined usage:
   python create_v2x_links.py --force --skip-existing  # 强制覆盖但跳过已存在(实际force会覆盖skip)
   python create_v2x_links.py --force --single "segment_name"  # 强制覆盖单个片段

8. 查看帮助 / View help:
   python create_v2x_links.py -h
   python create_v2x_links.py --help

在Python代码中直接使用 / Use directly in Python code:
=====================================================
    from create_v2x_links import V2XScenesLinkCreator
    
    # 创建链接创建器实例 / Create link creator instance
    creator = V2XScenesLinkCreator(
        source_dir="/home/myData/storage/code/HEAL/dataset/V2XScenes",
        target_dir="/home/myData/storage/code/V2XScenes/dataset/v2xscenes/data"
    )
    
    # 创建所有链接 / Create all links
    stats = creator.create_all_links(force=False)
    
    # 只创建单个链接 / Create only a single link
    creator.create_single_link("20240705_200812_300_1720181335_to_1720181350_15_39")
    
    # 列出所有片段 / List all segments
    creator.list_segments()
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Dict


class V2XScenesLinkCreator:
    """
    V2XScenes数据集软链接创建器
    V2XScenes dataset symbolic link creator
    """
    
    def __init__(self, base_source_dir: str, base_target_dir: str):
        """
        初始化 / Initialize
        
        Args:
            base_source_dir: 源文件根目录 / Source file root directory
            base_target_dir: 目标链接根目录 / Target link root directory
        """
        # 转换为绝对路径 / Convert to absolute path
        self.base_source_dir = Path(base_source_dir).resolve()
        self.base_target_dir = Path(base_target_dir).resolve()
        
        # 定义所有需要链接的片段 / Define all segments that need to be linked
        self.segments = [
            "20240712_111606_300_1720754431_to_1720754456_25_66",
            "20240709_084703_300_1720486033_to_1720486043_10_37",
            "20240705_200812_300_1720181335_to_1720181350_15_39",
            "20240705_203608_300_1720183229_to_1720183240_11_32",
            "20240706_195608_300_1720267127_to_1720267141_14_40",
            "20240709_075541_300_1720482952_to_1720482965_13_36",
            "20240709_075541_300_1720482986_to_1720482997_11_32",
            "20240709_082823_300_1720485060_to_1720485089_29_97",
            "20240709_082823_300_1720485180_to_1720485202_22_73",
            "20240709_083937_300_1720485599_to_1720485650_51_173",
            "20240709_172949_300_1720517416_to_1720517437_21_67",
            "20240709_172949_300_1720517585_to_1720517608_23_65",
            "20240711_205502_300_1720702651_to_1720702660_9_28",
            "20240711_205502_300_1720702788_to_1720702801_13_44",
            "20240711_211321_300_1720703639_to_1720703648_9_29",
            "20240711_211321_300_1720703803_to_1720703820_17_51",
            "20240712_105953_300_1720753203_to_1720753220_17_48",
            "20240712_111606_300_1720754206_to_1720754221_15_42",
            "20240712_111606_300_1720754254_to_1720754265_11_35",
            "20240712_111606_300_1720754373_to_1720754381_8_24"
        ]
    
    def create_symlink(self, source_path: Path, target_path: Path, force: bool = False) -> bool:
        """
        创建单个软链接 / Create a single symbolic link
        
        Args:
            source_path: 源路径 / Source path
            target_path: 目标链接路径 / Target link path
            force: 是否强制覆盖 / Whether to force overwrite
        
        Returns:
            bool: 成功返回True / Return True on success
        """
        # 检查源目录是否存在 / Check if source directory exists
        if not source_path.exists():
            print(f"  ✗ 源目录不存在 / Source directory does not exist: {source_path}")
            return False
        
        # 检查源路径是否为目录 / Check if source path is a directory
        if not source_path.is_dir():
            print(f"  ✗ 源路径不是目录 / Source path is not a directory: {source_path}")
            return False
        
        # 创建目标父目录 / Create target parent directory
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 处理已存在的路径 / Handle existing paths
        if target_path.exists():
            # 如果是软链接 / If it's a symbolic link
            if target_path.is_symlink():
                if force:
                    print(f"  → 移除已存在的软链接 / Removing existing symlink: {target_path}")
                    target_path.unlink()
                else:
                    print(f"  ⚠ 软链接已存在 / Symlink already exists: {target_path}")
                    response = input("    是否覆盖？/ Overwrite? (y/n): ")
                    if response.lower() == 'y':
                        target_path.unlink()
                    else:
                        print("    跳过此链接 / Skipping this link")
                        return False
            else:
                # 存在但不是软链接 / Exists but is not a symbolic link
                print(f"  ✗ 目标路径已存在且不是软链接 / Target path exists and is not a symlink: {target_path}")
                return False
        
        # 创建软链接 / Create symbolic link
        try:
            target_path.symlink_to(source_path)
            print(f"  ✓ 成功 / Success: {target_path.name} -> {source_path}")
            return True
        except Exception as e:
            print(f"  ✗ 创建失败 / Creation failed: {e}")
            return False
    
    def create_all_links(self, force: bool = False, skip_existing: bool = False) -> Dict[str, int]:
        """
        创建所有片段的软链接 / Create symbolic links for all segments
        
        Args:
            force: 是否强制覆盖已存在的软链接 / Whether to force overwrite existing symlinks
            skip_existing: 是否跳过已存在的链接 / Whether to skip existing links
        
        Returns:
            dict: 统计信息 / Statistics {'success': int, 'failed': int, 'skipped': int}
        """
        # 初始化统计信息 / Initialize statistics
        stats = {'success': 0, 'failed': 0, 'skipped': 0}
        
        print("=" * 80)
        print("开始创建V2XScenes数据集软链接 / Starting to create V2XScenes dataset symlinks")
        print(f"源根目录 / Source root: {self.base_source_dir}")
        print(f"目标根目录 / Target root: {self.base_target_dir}")
        print(f"共需处理 / Total segments to process: {len(self.segments)}")
        print("=" * 80)
        
        # 遍历所有片段 / Iterate through all segments
        for i, segment in enumerate(self.segments, 1):
            print(f"\n[{i}/{len(self.segments)}] 处理 / Processing: {segment}")
            
            # 构建源路径和目标路径 / Build source and target paths
            source_path = self.base_source_dir / segment
            target_path = self.base_target_dir / segment
            
            # 如果跳过已存在且目标存在 / If skip existing and target exists
            if skip_existing and target_path.exists():
                print(f"  ⊙ 已存在，跳过 / Already exists, skipping")
                stats['skipped'] += 1
                continue
            
            # 创建软链接 / Create symbolic link
            if self.create_symlink(source_path, target_path, force):
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        # 打印统计信息 / Print statistics
        print("\n" + "=" * 80)
        print("创建完成！统计信息 / Creation completed! Statistics:")
        print(f"  成功 / Success: {stats['success']}")
        print(f"  失败 / Failed: {stats['failed']}")
        print(f"  跳过 / Skipped: {stats['skipped']}")
        print(f"  总计 / Total: {len(self.segments)}")
        print("=" * 80)
        
        return stats
    
    def create_single_link(self, segment_name: str, force: bool = False) -> bool:
        """
        创建单个片段的软链接 / Create a symbolic link for a single segment
        
        Args:
            segment_name: 片段名称 / Segment name
            force: 是否强制覆盖 / Whether to force overwrite
        
        Returns:
            bool: 成功返回True / Return True on success
        """
        # 检查片段是否在列表中 / Check if segment is in the list
        if segment_name not in self.segments:
            print(f"错误 / Error: 片段 '{segment_name}' 不在列表中 / not in the list")
            print(f"可用片段 / Available segments: {', '.join(self.segments[:5])}...")
            return False
        
        # 构建源路径和目标路径 / Build source and target paths
        source_path = self.base_source_dir / segment_name
        target_path = self.base_target_dir / segment_name
        
        # 创建软链接 / Create symbolic link
        return self.create_symlink(source_path, target_path, force)
    
    def list_segments(self) -> None:
        """列出所有可用的片段 / List all available segments"""
        print("可用片段列表 / Available segments list:")
        print("-" * 80)
        for i, segment in enumerate(self.segments, 1):
            print(f"  {i:3d}. {segment}")
        print("-" * 80)
        print(f"总计 / Total: {len(self.segments)} 个片段 / segments")
    
    def get_segment_count(self) -> int:
        """
        获取片段总数 / Get total number of segments
        
        Returns:
            int: 片段总数 / Total number of segments
        """
        return len(self.segments)


def main():
    """主函数 / Main function"""
    # 默认路径配置 / Default path configuration
    default_source = "/home/myData/storage/code/HEAL/dataset/V2XScenes"
    default_target = "./dataset/v2xscenes/data"
    
    # 创建参数解析器 / Create argument parser
    parser = argparse.ArgumentParser(
        description='批量创建V2XScenes数据集软链接 / Batch create symbolic links for V2XScenes dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 / Examples:
  # 创建所有软链接 / Create all symbolic links
  %(prog)s
  
  # 强制覆盖已存在的软链接 / Force overwrite existing symbolic links
  %(prog)s --force
  
  # 跳过已存在的软链接 / Skip existing symbolic links
  %(prog)s --skip-existing
  
  # 只创建指定的片段 / Create only a specific segment
  %(prog)s --single "20240705_200812_300_1720181335_to_1720181350_15_39"
  
  # 列出所有可用片段 / List all available segments
  %(prog)s --list
  
  # 自定义源和目标路径 / Custom source and target paths
  %(prog)s --source /自定义/源路径 --target /自定义/目标路径
  %(prog)s --source /custom/source/path --target /custom/target/path
  
  # 组合使用 / Combined usage
  %(prog)s --force --single "segment_name"  # 强制覆盖单个片段 / Force overwrite single segment
        """
    )
    
    # 添加命令行参数 / Add command line arguments
    parser.add_argument('-s', '--source', default=default_source,
                       help=f'源文件根目录 / Source file root directory (默认 / Default: {default_source})')
    parser.add_argument('-t', '--target', default=default_target,
                       help=f'目标链接根目录 / Target link root directory (默认 / Default: {default_target})')
    parser.add_argument('-f', '--force', action='store_true',
                       help='强制覆盖已存在的软链接 / Force overwrite existing symbolic links')
    parser.add_argument('--skip-existing', action='store_true',
                       help='跳过已存在的软链接 / Skip existing symbolic links')
    parser.add_argument('--single', type=str, metavar='SEGMENT',
                       help='只创建指定的单个片段 / Create only a specific segment')
    parser.add_argument('--list', action='store_true',
                       help='列出所有可用的片段 / List all available segments')
    
    # 解析参数 / Parse arguments
    args = parser.parse_args()
    
    # 创建链接创建器实例 / Create link creator instance
    creator = V2XScenesLinkCreator(args.source, args.target)
    
    # 列出片段 / List segments
    if args.list:
        creator.list_segments()
        return
    
    # 创建单个链接 / Create single link
    if args.single:
        success = creator.create_single_link(args.single, args.force)
        sys.exit(0 if success else 1)
    
    # 创建所有链接 / Create all links
    stats = creator.create_all_links(
        force=args.force,
        skip_existing=args.skip_existing
    )
    
    # 根据失败数量决定退出码 / Determine exit code based on number of failures
    sys.exit(0 if stats['failed'] == 0 else 1)


if __name__ == "__main__":
    main()