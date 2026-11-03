# NDT 虚拟样本生成项目

根据已有数值型回归数据生成虚拟样本。默认示例使用 66 条原始记录，生成 100 条新记录，以 `TC(W/mK)` 为连续目标、`Saturated` 为类别特征，并排除编号 `No.`。目标列是本项目运行示例的选择，可以通过参数修改。

## 快速开始（Windows PowerShell）

先安装 Python 3.10 或更新版本，解压项目，在包含本文件的目录打开终端：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ndt_samples --output generated/first_run.csv
```

结果为 `generated/first_run.csv`，只包含虚拟记录。交付包另附已生成的 `generated/virtual_samples.csv` 示例。再次运行时需指定新路径：

```powershell
.\.venv\Scripts\python.exe -m ndt_samples --n-virtual 500 --seed 42 --output generated/virtual_500.csv
```

程序不会覆盖已有结果，也不会覆盖输入文件。如果使用已有的 Python 环境，可直接运行 `python -m pip install -r requirements.txt` 和 `python -m ndt_samples --output generated/new_samples.csv`。

## 自定义数据

支持带表头的 CSV 或 TSV，自动检测分隔符。所有参与生成的列应为有限数值，不接受缺失值、无穷值及文本类别。请预先把类别编码成数字，并明确指定类别列。`y` 为连续回归目标，不用于分类标签。

```powershell
python -m ndt_samples --input data/samples.tsv --target "TC(W/mK)" --categorical Saturated --drop No. --n-virtual 100 --noise-ratio 0.01 --seed 42 --output generated/run2.csv
```

- `--target`：目标列，默认 `TC(W/mK)`。
- `--categorical`：一个或多个数字编码类别列，默认 `Saturated`；如果没有类别列，单独写 `--categorical`。
- `--drop`：不参与生成的列，默认 `No.`；如果没有需要排除的列，单独写 `--drop`。
- `--n-virtual`：生成数量，允许 0。
- `--noise-ratio`：非负噪声比例，默认 0.01。
- `--seed`：随机种子，默认 42；相同输入、参数及依赖环境可复现。
- `--output`：新结果路径，已有文件会被拒绝覆盖。

比如改为预测硬度：`--target "Hardness(HRC)"`。此时热导率会成为输入特征，是否适合实际应用应根据预测时可获得的测量量决定。

## Python 调用

从项目根目录运行：

```python
import pandas as pd
from ndt_samples import generate_virtual_samples

data = pd.read_csv('data/samples.tsv', sep='\t')
X = data.drop(columns=['No.', 'TC(W/mK)'])
y = data['TC(W/mK)']
X_virtual, y_virtual = generate_virtual_samples(
    X, y, n_virtual=100, noise_ratio=0.01,
    categorical_columns=['Saturated'], random_state=42,
)
```

接受 DataFrame 或二维数值数组；数组的类别列使用整数索引。目标接受一维数组、Series 或单列 DataFrame，按行位置对应，不自动按索引对齐。DataFrame 输入保留列名，并保留类别列的数据类型。

## 算法及修复

1. 按观测到的类别组合分组，按原始组频率随机选择生成组；不跨饱和状态混合。
2. 在组内将连续特征标准化后计算欧氏距离，以逆密度概率选择基础样本。
3. 从其他样本中选择最多两条邻居，使用 Dirichlet 权重插值连续特征与目标。组内只有一行时，输出该行的复制。
4. 根据标准化协方差矩阵的特征分解，统一生成相关连续噪声，再还原尺度；常量列不加噪声。
5. 将连续值与目标限制在各组原始最小、最大值之间，类别值保持不变。

修复了原脚本缺少导入、单特征相关矩阵报错、单行抽样失败、二维单列目标报错、空输出维度错误，以及类别变小数等问题。NumPy 可直接完成距离计算，因此无需 scikit-learn。此版本更改了采样策略，不会复现原脚本的数值输出。

范围裁剪和类别约束不等同于完整的物理约束：项目不会自动验证化学成分总和、测量单位或数据标注。输入数据保留原值，使用前请核实其科学含义。虚拟目标来自原始目标的插值，不能视为新的实验测量，也不能保证提高模型性能。

训练评估时，请先按真实样本划分训练集和测试集，仅对训练集生成虚拟样本。交叉验证时每折分别生成；若干湿记录来自同一试件，应按试件分组划分，避免关联记录泄漏。全量生成命令只演示生成与导出，不执行模型评估。

组内距离矩阵的时间和内存开销约为 O(n²)，适用于本例的小样本数据；大数据需改用分块距离或近邻索引。

## 验证

```powershell
python -m unittest discover -s tests -v
```

7 项测试覆盖单特征、单列目标、单行和常量、空输出、类别与分组边界、随机种子、全类别输入和无效输入。交付时在 Python 3.10.18、NumPy 1.23.5、pandas 2.3.2 环境通过，并使用真实数据完成 CLI 生成。依赖范围未逐版本测试。

## 文件

```text
ndt_samples/       可导入模块及命令行入口
data/samples.tsv   原 DATA 的表格内容，去除了首行说明 # DATA
tests/             unittest 测试
generated/         已生成的 100 条示例及验证摘要
requirements.txt   NumPy、pandas 依赖
```

## 来源

原仓库：https://github.com/gelatiao/NDT-virtual-sample-generation-framework

- 原代码文件：`def generate virtual sample`，blob SHA `4851c9742898659773b5dded58fd20cc499cc23b`。
- 原数据文件：`DATA`，blob SHA `04f328ace8155bf0d24faaebc0ab56d43433bad3`。
- 本项目由原始脚本整理为可运行模块；根目录原始 `DATA` 和 `def generate virtual sample` 保留作参考，运行请使用 `ndt_samples`。原仓库根目录未包含许可证文件，未擅自补充授权条款。
