#!/bin/bash

# 定义源目录和目标目录
SOURCE_DIR="datasets"
TARGET_DIR="/data/baiyutao/datasets"

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误：当前目录下不存在 '$SOURCE_DIR' 文件夹"
    exit 1
fi

# 检查目标目录是否存在，如果不存在则创建
if [ ! -d "/data/baiyutao" ]; then
    sudo mkdir -p "/data/baiyutao"
    echo "已创建目录 /data/baiyutao"
fi

# 移动文件夹
echo "正在移动 '$SOURCE_DIR' 到 '$TARGET_DIR'..."
mv "$SOURCE_DIR" "$TARGET_DIR"

# 检查移动是否成功
if [ $? -ne 0 ]; then
    echo "移动文件夹失败"
    exit 1
fi

# 创建软链接
echo "正在创建软链接 '$SOURCE_DIR' 指向 '$TARGET_DIR'..."
ln -s "$TARGET_DIR" "$SOURCE_DIR"

# 检查软链接是否创建成功
if [ $? -eq 0 ]; then
    echo "操作完成！原 '$SOURCE_DIR' 现在是软链接，指向 '$TARGET_DIR'"
else
    echo "创建软链接失败"
    exit 1
fi