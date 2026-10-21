# DASNA

**面向图异常检测的扩散增广与选择性邻域聚合**

[English](README.md)

DASNA 面向标签不平衡的监督式节点异常检测。整体流程分为三个部分：在训练子图上拟合扩散生成器，离线生成异常节点；构造并筛选邻居候选；通过多分支 Mamba 聚合得到节点表示和异常分数。

![DASNA 框架](docs/figures/framework.png)


## 目录

| 路径 | 内容 |
| --- | --- |
| `dasna/augmentation/` | 模块 A，图自编码器与扩散增广 |
| `dasna/detection/` | 模块 B、C，邻居筛选、Mamba 聚合与分类训练 |
| `dasna/preprocessing.py` | 原始图的邻接表预处理 |
| `configs/` | 六个数据集的配置模板 |
| `environments/` | 两套环境的依赖与版本记录 |
| `data/raw/` | 原始 MAT 和历史特征缓存，仅本地保存 |
| `data/processed/` | 邻接表缓存，仅本地保存 |
| `data/generated/` | 独立增广图，仅本地保存 |
| `artifacts/` | 训练输出，运行时创建且不上传 |
| `docs/` | 摘要、框架图、环境和整理说明 |
| `scripts/` | 环境检查、数据清单和上传文件检查 |

## 运行环境


- `dasna-augmentation`：扩散增广；Python 3.10、PyTorch 2.6.0+cu124。本机 RTX 5060 Ti 与该版本 CUDA 内核不兼容，当前使用 CPU，即 `--device -1`。
- `dasna-detection`：检测；Python 3.10、PyTorch 2.8.0+cu128，已验证 Mamba 的 CUDA 前向运行。

新环境安装步骤和实测限制见 [环境说明](docs/environment.md)。以下命令在项目根目录执行：

```bash
conda activate dasna-augmentation
python scripts/check_environment.py --stage augmentation --device cpu

conda activate dasna-detection
python scripts/check_environment.py --stage detection --device cuda
```

## 数据与运行入口

新克隆项目需要另外准备带有固定训练、验证、测试 mask 的数据文件，具体见 [数据说明](data/README.md)。

```bash
conda activate dasna-detection
python -m dasna.preprocessing --datasets amazon

conda activate dasna-augmentation
python -m dasna.augmentation.train --dataset amazon --ratio 10 --device -1 --output-dir data/generated/train-ratio

conda activate dasna-detection
python -m dasna.detection.train --config configs/amazon.yaml --generated-dir data/generated/train-ratio
```
