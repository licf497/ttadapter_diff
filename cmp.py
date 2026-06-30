#!/usr/bin/env python3
"""
扫描文件夹中所有 .ttadapter 文件，去掉 loc 信息后，
提取每个 kernel (func.func)，计算 sha256，输出 Excel。

用法:
  python3 hash_ttadapters.py <文件夹路径> [输出xlsx路径]

示例:
  python3 hash_ttadapters.py /path/to/ttadapters/
  python3 hash_ttadapters.py /path/to/ttadapters/ output.xlsx
"""

import sys
import os
import hashlib
import json
from openpyxl import Workbook


def strip_loc(text: str) -> str:
    """去除所有 #loc 定义行和 loc(...) 引用（处理嵌套括号）"""
    # 1. 去掉 #loc 定义行
    lines = text.split('\n')
    lines = [l for l in lines if not l.strip().startswith('#loc')]
    text = '\n'.join(lines)

    # 2. 去掉 loc(...)  —— 处理嵌套括号
    result = []
    i = 0
    while i < len(text):
        if text[i:i+4] == 'loc(':
            depth = 1
            j = i + 4
            while j < len(text) and depth > 0:
                if text[j] == '(':
                    depth += 1
                elif text[j] == ')':
                    depth -= 1
                j += 1
            i = j
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


def extract_kernels(text: str) -> list[tuple[str, str]]:
    """
    从 MLIR 文本中提取所有 func.func。
    返回 [(kernel_name, kernel_text), ...]
    """
    kernels = []
    idx = 0
    while True:
        # 找 func.func @name
        pos = text.find('func.func @', idx)
        if pos == -1:
            break

        # 提取 kernel 名
        name_start = pos + len('func.func @')
        name_end = name_start
        while name_end < len(text) and text[name_end] not in ('(', '\n', ' '):
            name_end += 1
        kernel_name = text[name_start:name_end].strip()

        # 从 func.func 开始找 body 的起始 {
        # 结构: func.func @name(...) [attributes {...}] { body }
        # 先找到 attributes 块的结束，再找 body
        cur = pos

        # 跳过参数列表 (...)
        paren_depth = 0
        arg_started = False
        while cur < len(text):
            c = text[cur]
            if c == '(':
                paren_depth += 1
                arg_started = True
            elif c == ')':
                paren_depth -= 1
            elif c == '{' and paren_depth == 0:
                # 找到了 attributes 块或 body
                break
            cur += 1
            if arg_started and paren_depth == 0:
                # 参数列表结束
                break

        # 跳过空白和 attributes {...}
        while cur < len(text):
            c = text[cur]
            if c == '{':
                # 检查前面是否有 attributes 关键字
                before = text[pos:cur].strip()
                # 找到最后一个单词
                parts = before.split()
                if parts and parts[-1] == 'attributes':
                    # 这是 attributes 块，跳过
                    brace_depth = 1
                    cur += 1
                    while cur < len(text) and brace_depth > 0:
                        if text[cur] == '{':
                            brace_depth += 1
                        elif text[cur] == '}':
                            brace_depth -= 1
                        cur += 1
                    continue
                else:
                    # 这是 body 块
                    body_start = cur
                    brace_depth = 1
                    cur += 1
                    while cur < len(text) and brace_depth > 0:
                        if text[cur] == '{':
                            brace_depth += 1
                        elif text[cur] == '}':
                            brace_depth -= 1
                        cur += 1
                    body_end = cur
                    kernel_text = text[pos:body_end]
                    kernels.append((kernel_name, kernel_text))
                    break
            cur += 1

        idx = body_end if kernels else pos + 1

    return kernels


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def main():
    if len(sys.argv) < 2:
        print("用法: python3 hash_ttadapters.py <文件夹路径> [输出xlsx路径]")
        sys.exit(1)

    folder = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'kernel_hashes.xlsx'

    if not os.path.isdir(folder):
        print(f"错误: '{folder}' 不是有效文件夹")
        sys.exit(1)

    # 递归收集所有 ttadapter 文件
    files = []
    for root, dirs, filenames in os.walk(folder):
        for f in filenames:
            if f.endswith('.ttadapter'):
                files.append(os.path.join(root, f))
    files.sort()
    if not files:
        print("未找到 .ttadapter 文件")
        sys.exit(1)

    print(f"找到 {len(files)} 个 ttadapter 文件")

    # 处理每个文件
    rows = []  # [(filename, kernel_name, sha256), ...]
    for filepath in files:
        filename = os.path.relpath(filepath, folder)
        with open(filepath, 'r') as f:
            raw = f.read()

        clean = strip_loc(raw)
        kernels = extract_kernels(clean)

        if not kernels:
            print(f"  警告: {filename} 中未找到 func.func")
            continue

        for kname, ktext in kernels:
            h = sha256(ktext)
            rows.append((filename, kname, h))
            print(f"  {filename}  @{kname}  {h[:16]}...")

    # 生成 Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "Kernel Hashes"
    ws.append(["TTAdapter File", "Kernel Name", "SHA256"])

    for row in rows:
        ws.append(list(row))

    # 调整列宽
    ws.column_dimensions['A'].width = 50
    ws.column_dimensions['B'].width = 40
    ws.column_dimensions['C'].width = 70

    wb.save(output_path)
    print(f"\n共 {len(rows)} 个 kernel，输出: {output_path}")


if __name__ == '__main__':
    main()