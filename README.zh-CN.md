# DASNA

**面向图异常检测的扩散增广与选择性邻域聚合**

[English](README.md)

DASNA 面向标签不平衡的监督式节点异常检测。整体流程分为三个部分：在训练子图上拟合扩散生成器，离线生成异常节点；构造并筛选邻居候选；通过多分支 Mamba 聚合得到节点表示和异常分数。

![DASNA 框架](docs/figures/framework.png)

原始摘要和框架 PDF 位于 [docs/](docs/)。摘要中的实验结论属于作者提供的内容，本次文件整理未重新验证。

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

发布文档使用以下环境示例名；本机已有 Conda 环境未改动。复用已有环境时，将命令中的环境名替换为实际名称。以下版本来自整理时两套已有环境的实测：

- `dasna-augmentation`：扩散增广；Python 3.10、PyTorch 2.6.0+cu124。本机 RTX 5060 Ti 与该版本 CUDA 内核不兼容，当前使用 CPU，即 `--device -1`。
- `dasna-detection`：检测；Python 3.10、PyTorch 2.8.0+cu128，已验证 Mamba 的 CUDA 前向运行。

Mamba 是本机安装的特定构建，不能只凭版本号保证另一台 Windows 机器直接重建。新环境安装步骤和实测限制见 [环境说明](docs/environment.md)。以下命令在项目根目录执行：

```bash
conda activate dasna-augmentation
python scripts/check_environment.py --stage augmentation --device cpu

conda activate dasna-detection
python scripts/check_environment.py --stage detection --device cuda
```

## 数据与运行入口

本机数据已保留在新目录，Git 会排除 MAT、邻接表和生成样本。新克隆项目需要另外准备带有固定训练、验证、测试 mask 的数据文件，具体见 [数据说明](data/README.md)。

```bash
conda activate dasna-detection
python -m dasna.preprocessing --datasets amazon

conda activate dasna-augmentation
python -m dasna.augmentation.train --dataset amazon --ratio 10 --device -1 --output-dir data/generated/train-ratio

conda activate dasna-detection
python -m dasna.detection.train --config configs/amazon.yaml --generated-dir data/generated/train-ratio
```

这个例子把新样本写入单独目录，保留历史缓存。已有新格式检查点时，用 `python -m dasna.augmentation.generate` 代替训练入口即可只生成样本；各入口均支持 `--help`。生成器默认拒绝覆盖已有样本。

六份 YAML 是运行模板，没有冒充论文最终参数。开始训练前请检查增广开关、样本后缀、设备和特征维度。

新生成器按训练集异常节点数量计算增广比例，输出仅包含异常节点的独立训练图。原来的八份增广缓存按全图异常节点数量计算比例，文件未改动；不能把两种定义混用于实验。生成节点只加入训练集合，评估时关闭正类邻居补充。

## 上传内容与验证范围

适合纳入 GitHub 的内容包括代码、配置模板、环境文件、中英文说明、摘要、框架图和必要的第三方声明。原始大数据、缓存、模型、日志和旧嵌套 Git 历史不纳入发布。

```bash
python scripts/check_repository.py
```

本次验证覆盖入口、环境、小型模型操作和数据转换，没有重新运行完整论文实验。原始筛选实现中需要进一步核对的方法问题、缓存比例变化和检查点兼容变化记在 [整理说明](docs/release-notes.md)。

两组可重跑测试的命令和实测范围见 [验证记录](docs/validation.md)。

旧项目宣传、徽章和发布元数据已清理，必要的来源与许可证信息集中保存在 [第三方声明](THIRD_PARTY_NOTICES.md) 和 `LICENSES/`。其中未确认再分发授权的组件需要在公开发布前核实；本次没有对整个项目擅自指定统一开源许可证。
