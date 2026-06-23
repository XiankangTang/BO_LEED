#!/bin/bash

# 方法1：使用find命令（推荐）
find . -type f -name "*_xy.png" -delete

# 或者方法2：使用find命令并打印删除的文件
# find . -type f -name "*_xy.png" -print -delete

# 或者方法3：使用循环（如果需要更复杂的操作）
# find . -type f -name "*_xy.png" | while read -r file; do
#     echo "删除文件: $file"
#     rm -f "$file"
# done

# 或者方法4：使用find的-exec参数
# find . -type f -name "*_xy.png" -exec rm -f {} \;
