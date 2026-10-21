# 环境说明

项目分别使用增强与检测两套 Conda 环境。发布文档中的示例名为
`dasna-augmentation` 和 `dasna-detection`，没有据此重命名本机环境。
2026-10-21 对已有环境进行了只读检查，未安装、升级或修改其中的包。
复用已有环境时，将 `conda activate` 后的示例名替换为对应的实际环境名。

## 文件的用途

- `environments/augmentation.yml`、`environments/detection.yml` 仅创建 Python 3.10 和 pip 的基础环境。
- `requirements.augmentation.txt`、`requirements.detection.txt` 固定相关普通包版本。先选择 PyTorch
  的 CPU/CUDA 构建，再安装这些 requirements；Mamba 与图分区扩展另行安装。
- `observed.augmentation.txt`、`observed.detection.txt` 是已脱敏的相关依赖快照，保留实际 CUDA 构建
  版本。它们不是带哈希的完整锁文件，也不包含完整 Conda 环境、驱动或编译工具链。

这些文件提供重建起点，尚未在一台全新机器上完成安装验证。兼容的已有增强和检测环境
可直接复用；请勿为了读取本项目而覆盖已有环境。若需独立验证，在 `conda env create`
命令中加 `-n` 指定新的环境名。

## 已实测的关键版本与能力

| 项目 | 检测环境（分类） | 增强环境（生成） |
| --- | --- | --- |
| Python | 3.10.20 | 3.10.20 |
| PyTorch | 2.8.0+cu128 | 2.6.0+cu124 |
| NumPy / SciPy | 2.2.6 / 1.15.3 | 2.2.6 / 1.15.3 |
| scikit-learn | 1.7.2 | 1.7.2 |
| PyYAML | 6.0.3 | 未安装，生成入口不依赖 |
| torch-geometric | 2.3.1（分类入口不依赖） | 2.7.0 |
| pygod | 1.1.0（分类入口不依赖） | 1.1.0 |
| Mamba | 2.2.3.post1，本地源码安装 | 未安装 |
| causal-conv1d | 1.4.0 | 未安装 |

两环境 `pip check` 均通过。已有分类模块及 Mamba 可导入，`MAMBA_AVAILABLE=True`；
检测环境中 CUDA 张量加法以及小型 Mamba CUDA 前向通过。已有生成模块、DGL 和 PyG 扩展
可导入；增强环境中 CPU 张量加法、4 节点图的 `ClusterData` 二分通过。

当前 GPU 是 RTX 5060 Ti（计算能力 `sm_120`）。增强环境的 PyTorch 2.6.0+cu124 虽然
报告 `torch.cuda.is_available() == True`，实际 CUDA 张量运算失败，报错
`no kernel image is available for execution on the device`；它所列支持架构最高为
`sm_90`。因此，此环境在这块 GPU 上应使用 `--device -1`（生成入口默认即为 CPU）。
要启用该 GPU，需要另建与硬件相容的 PyTorch/CUDA/PyG 扩展组合并重新验证；本轮未
执行这项环境迁移。GPU 可被发现不等于计算内核可运行。

上述检查不包含训练、完整数据流程、性能评测或论文结果复现。

## 新建生成环境的参考步骤

以下从仓库根目录执行，创建基础环境后安装 CPU 组合，避免把当前机器上不兼容的
CUDA 12.4 构建当作通用配置。此安装步骤未在全新环境实跑。

```powershell
conda env create -f environments/augmentation.yml
conda activate dasna-augmentation
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r environments/requirements.augmentation.txt
python -m pip install --no-index --only-binary=:all: torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
python -m pip check
```

`ClusterData` 的图分区需要匹配 PyTorch 构建的原生扩展。本地通过的是
`torch-scatter 2.1.2+pt26cu124` / `torch-sparse 0.6.18+pt26cu124`；上面的 CPU
组合是重建建议，不等同于已测试的 CUDA wheel。若平台没有匹配 wheel，命令会停止，
需要按 [PyG 官方安装文档](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
选择受支持组合。DGL 只涉及原始数据获取；当前 MAT 预处理流程不要求安装 DGL。

## 新建分类环境与 Mamba 限制

本机检测环境的 CUDA 12.8 基础组合已通过小型运算检查。重建时可先准备普通依赖：

```powershell
conda env create -f environments/detection.yml
conda activate dasna-detection
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environments/requirements.detection.txt
python -m pip check
```

这一步尚未安装 Mamba，不能直接运行启用 Mamba 的检测配置。本地 Mamba 来自一个
本机源码目录，未确认可分享的 wheel、源码提交或本地补丁；因此不能保证
`pip install mamba-ssm==2.2.3.post1` 重建相同环境。已观察到的配套版本为
`causal-conv1d==1.4.0`、`einops==0.8.2`、`transformers==4.36.2`。

在新机器上，需按 [Mamba 官方源码说明](https://github.com/state-spaces/mamba)
准备匹配操作系统、PyTorch、CUDA 与 GPU 架构的构建，并验证一个实际前向计算。
若要交付完全可重建的 Windows 环境，还需补齐原安装所用源码提交、修改记录、构建
步骤和可合法分发的安装产物。此仓库没有捆绑本机环境或这些未知产物。

PyTorch 构建的选择以 [官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)
为准；`requirements` 中不带 CUDA 后缀的版本可匹配预先安装的相应 CPU/CUDA 构建。
