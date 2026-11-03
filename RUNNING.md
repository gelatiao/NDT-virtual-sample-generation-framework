# 六模型主程序运行说明

本程序以本地 TC 目录中的 MLP、SVR、RF、GBDT、CatBoost、XGBoost 六个脚本为主要实现来源，保留它们逐步增加虚拟样本、重新训练、对比真实训练/测试集的流程。论文 DOI：10.1016/j.measurement.2025.118248。

## 安装与运行

在 Python 3.10 或更新版本中，从项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-models.txt
.\.venv\Scripts\python.exe run_experiment.py --model mlp --output runs/mlp_local
```

输出目录必须是新目录，避免覆盖既有实验。运行六种模型：

```powershell
python run_experiment.py --model all --config configs/local.json --output runs/six_models_local
```

先用 100 条总训练记录检查全流程：

```powershell
python run_experiment.py --model all --max-samples 100 --output runs/smoke
```

`--max-samples` 指真实训练行加虚拟行的总数，不是新增数量；如小于实际训练行数，仅运行基线。默认每步 25 条，最后一步补齐至总数上限，不会超出。local.json 默认上限 800；这是按本地 target_size 限制总训练数的语义，不是声称论文“800”的所有描述都无歧义。

## 两种明确标记的流程

### 本地流程 local.json

- 使用 local_sources.json 指定的本地函数，函数体原样提取（仅重命名）；导入不会执行原脚本顶层训练与绘图。
- MLP 保留三样本 Dirichlet 插值加均匀噪声的简化生成器；其余五种保留密度加相关噪声生成器。生成发生在标准化后的训练特征空间，RF 在筛选后的多项式空间，后续步骤仍从累积的增强集合生成。这些是本地行为，不等同于类别约束的改进生成器。
- 保留随机种子 42 的 8:2 随机划分；每个步骤用相同模型随机种子，生成器使用可追踪的逐步骤种子。
- 非 RF 使用本地均值 ±4 标准差的极端值边界处理；RF 保留 IQR 分位数/中位数处理、多项式、相关性筛选和 SelectKBest(6)。所有拟合都移动到训练集内，测试目标不被替换。
- 配置采用论文 Table 2 与本地搜索范围/摘要的参数。MLP 未在论文明确的 alpha=0.001、learning_rate_init=0.001 来自本地候选，不能声称已确认它们是产生发表结果的参数。保留本地增强模型 alpha×10，可在配置中设为 1。
- CatBoost 显式使用 Bernoulli bootstrap，使本地 subsample=0.9 参数可用；这是兼容设置，未确认原环境默认 bootstrap。

这是**以本地脚本为依据的工程整理版本**。它使用已核对的补充表，纠正了全量预处理泄漏、测试集选最好步骤、负 R² 过滤和模拟预测问题，没有复刻旧脚本的数据目标替换、整个网格搜索、绘图编辑或早停。固定参数逐步训练到上限，结果不会承诺等于论文指标。

### 改进评估 grouped.json

```powershell
python run_experiment.py --model all --config configs/grouped.json --output runs/six_models_grouped
```

使用化学成分组合代理试件组进行 5 折验证；真实试件 ID 尚未提供，不把该代理称为已核实试件身份。使用目前的改进生成器，在真实训练原始特征上按饱和状态分组生成，之后通过该折已拟合的预处理器转换。开启氧化物行总和归一化至 100%，MLP alpha 不再同时改变。

本版本使用**预先固定的模型与增强参数**，不执行嵌套调参，也不按测试成绩挑选最佳样本量。各步骤测试曲线属于多配置探索结果，若后续根据曲线调参，需另设独立测试集或接入内层搜索再报告最终性能。

可在 JSON 中更改 seeds、step_size、max_samples、noise_ratio、normalize_oxides 和模型参数；完整生效配置都会保存。`--protocol`、`--generator` 可以覆盖配置，但改变它们就是新实验。

## 输出

- metrics.csv：每模型/种子/折/步骤的真实训练与真实测试 R²、RMSE、MAE；负 R² 保留。
- predictions.csv：上述每一步逐条真实目标和预测，带样本 ID；虚拟数据不参与测试。
- config.json：生效参数、依赖版本、输入文件和当前运行模块 SHA256。
- summary.json：按模型和步骤平均的测试指标；各折 R² 平均不是合并折外 R²，字段含义为 mean_test_metrics_by_step。
- warnings.json：模型训练警告（例如未收敛），不静默隐藏。
- failures.json：失败的模型/折与 traceback；失败时程序返回非零，其他已完成折的结果仍保存。

不创建“最佳测试集模型”或手工填写指标。当前不保存模型二进制，不直接提供部署推理；需要重新执行相同配置训练。各模型的结果均来自实际 predict。

## 来源与待补充功能

`configs/local_sources.json` 保存六份本地源文件名称、SHA256 与生成器行号，说明迁移来源。已有 `python -m ndt_samples` 数据生成命令仍可使用；evaluate.py 的岭回归是独立扩展基线。

当前未迁移全量网格搜索、SHAP、GP/LightGBM、CNN 和论文全部图表。接入这些功能前，应先用此入口建立可追溯基线，避免把历史图片反推的数据当作真实训练记录。

运行测试：`python -m unittest discover -s tests -v`。
