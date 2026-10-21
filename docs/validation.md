# 本地验证记录

验证日期：2026-10-21。以下检查在既有 Conda 环境中运行，没有安装、升级或修改环境。
下方激活命令使用发布文档中的环境示例名，复用已有环境时需替换为实际名称。

## 环境能力

```powershell
conda activate dasna-augmentation
python scripts/check_environment.py --stage augmentation --device cpu

conda activate dasna-detection
python scripts/check_environment.py --stage detection --device cuda
```

生成端 CPU 张量运算和 METIS 小图分区通过；检测端 CUDA 张量运算与 Mamba 小型前向
通过。生成端当前 PyTorch 构建与本机 GPU 不兼容，具体版本及限制见
[environment.md](environment.md)。两环境的 `pip check` 通过。

## 可重复的小型回归测试

分别在两个环境运行，不能把两套测试误当作单一环境下的完整训练。

```powershell
conda activate dasna-augmentation
python -m unittest discover -s tests -p test_augmentation_pipeline.py -v

conda activate dasna-detection
python -m unittest discover -s tests -p test_detection_pipeline.py -v
```

| 测试组 | 数量 | 覆盖内容 |
| --- | ---: | --- |
| 扩散增广集成 | 2 | 独立进程训练与生成、跨入口状态检查点、训练集归一化和比例、分批索引、独立异常训练图、拒绝覆盖、原始文件不变、稀疏图与无边分区 |
| 检测与预处理 | 6 | 真实关系图保留、非连续 mask 重映射、仅训练集增广、单节点及整除批次、检查点重载、评估模式与无梯度、路径解析、CPU 下明确拒绝 Mamba |

测试使用临时小型数据和短训练轮次，并自动清理测试数据与检查点。它们验证运行接口与
数据处理约束，不验证六个数据集的模型性能。上述检测回归测试采用显式 GNN CPU
路径；真实 CUDA Mamba 的额外小型训练检查见下方上传前复查记录。没有执行完整数据
上的 Mamba 训练。

四个模块命令的 `--help` 均通过：`dasna.preprocessing`、
`dasna.augmentation.train`、`dasna.augmentation.generate`、`dasna.detection.train`。

## 文件与打包

- 124 份原始代码、文档与数据文件和外部备份的 SHA-256 一致；可变 Git 索引与
  Python 字节码不作为源码备份一致性依据。
- 99 份迁入新数据/文档目录的文件与原文件 SHA-256 一致。
- 原始六份 MAT 与八份增广 MAT 的结构、标签、mask 和摘要哈希已写入
  [data/inventory.json](../data/inventory.json)。
- 临时构建的 wheel 和源码包均实际包含第三方声明、Apache/PyG 许可证和 TabSyn
  NOTICE，未包含数据、权重或缓存；未给整个项目指定统一开源许可证。
- `python scripts/check_repository.py` 检查待分享文件的 Python 语法、必要文件、
  大小及是否混入数据、检查点或缓存。GitHub 工作流执行同一轻量检查。

完整实验、最终论文参数、基线对照与结果排名均未重新验证。已知方法问题和旧文件
兼容变化见 [release-notes.md](release-notes.md)。

## 上传前复查

2026-10-21 再次执行两组回归测试，增强 2 项、检测 6 项全部通过；生成端
CPU/METIS 和检测端 CUDA/Mamba 环境检查再次通过。

额外执行真实 CUDA Mamba 小型集成：12 个原始节点、2 个选中的增广训练节点、
4 维特征和三个关系分支，启用 Mamba、residual 和增广后训练 1 个 epoch。
15 次分支前向、3 次优化器更新通过，Mamba 梯度均有限且非零；检查点保存和
重载成功，验证与测试均使用 eval/no_grad，输出指标均为有限值。该检查没有
评价实际检测性能，也不替代完整数据训练和已记录的 residual 计算核对。

发布摘要保留为 Markdown，带有个人编辑元数据的 Word 源文件留在本地排除目录，
不纳入上传。最终文件检查覆盖名称清理、本机路径和凭据标记、相对链接、数据与
缓存排除，以及 ZIP 和 Git 暂存内容一致性。该检查不替代方法公式核对、完整实验
复现或尚未确认的第三方授权。
