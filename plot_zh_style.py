#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Word / 论文插图统一风格：画布尺寸、导出 DPI、中英文字号；中文优先黑体（SimHei）。
供 plot_*_group_zh.py 共用，避免四份脚本字号不一致。
"""
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

# 画布（英寸）— 四脚本一致
FIG_SIZE = (12.0, 7.0)

# 导出 PNG 分辨率（插入 Word 经缩放后仍较清晰）
SAVE_DPI = 400

FONT_SIZE_TITLE = 28
FONT_SIZE_LABEL = 26
FONT_SIZE_TICK = 26
FONT_SIZE_LEGEND = 26


def configure_matplotlib_fonts():
    """中文黑体（SimHei）为主，英文/数字可回退 Times New Roman；并设置坐标轴/图例默认字号。"""
    rcParams["font.family"] = ["SimHei", "Times New Roman"]
    rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "SimSun",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Times New Roman",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    rcParams["axes.unicode_minus"] = False
    rcParams["font.size"] = FONT_SIZE_TICK
    rcParams["axes.titlesize"] = FONT_SIZE_TITLE
    rcParams["axes.labelsize"] = FONT_SIZE_LABEL
    rcParams["xtick.labelsize"] = FONT_SIZE_TICK
    rcParams["ytick.labelsize"] = FONT_SIZE_TICK
    rcParams["legend.fontsize"] = FONT_SIZE_LEGEND
    rcParams["savefig.dpi"] = SAVE_DPI


def chinese_label_font():
    """轴标签用黑体（Windows 为 SimHei）。"""
    return FontProperties(family="SimHei", size=FONT_SIZE_LABEL)


def chinese_legend_font():
    """图例用黑体，字号略小于轴标签。"""
    return FontProperties(family="SimHei", size=FONT_SIZE_LEGEND)


def savefig_kw():
    """统一 savefig 参数。"""
    return {
        "dpi": SAVE_DPI,
        "bbox_inches": "tight",
        "facecolor": "white",
        "edgecolor": "none",
        "pad_inches": 0.08,
    }
