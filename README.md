# NDT 虚拟样本生成与热导率预测

Hu 等（2025）论文配套项目的可运行整理版本。论文：An integrated approach for thermal conductivity prediction of grotto sandstone using virtual sample generation and machine learning with limited nondestructive testing data，Measurement 256，118248。DOI：https://doi.org/10.1016/j.measurement.2025.118248。

主要运行程序以本地 TC 项目的六种模型脚本为依据，统一渐进增强、训练、真实预测和指标导出。根目录原始 DATA 和旧函数文件保留作历史参考。

## 安装与运行

Python 3.10 或更新版本，在项目根目录执行（建议先创建虚拟环境）：

```powershell
python -m pip install -r requirements-models.txt
python run_experiment.py --model mlp --output runs/mlp_local
```

六种模型快速运行（总训练记录上限为 100）：

```powershell
python run_experiment.py --model all --max-samples 100 --output runs/six_models
```

支持 mlp、svr、rf、gbdt、catboost、xgboost。默认逐步新增 25 条，最大总训练记录数 800。输出目录必须不存在，避免覆盖历史结果。完整说明见 [RUNNING.md](RUNNING.md)。

## 输入与结果

`data/samples.tsv` 的 66 条记录与论文补充数据表逐值一致。原始数据保留三位小数；不自动替换目标。字段与来源说明见 [data/README.md](data/README.md)。

每次运行保存 metrics.csv、predictions.csv、comparisons.csv、config.json、summary.json、warnings.json 和 failures.json。逐条预测包含真实目标和样本编号，指标全部来自实际模型 predict；负 R² 不删除。comparisons.csv 比较预先指定的最终步骤与基线，不挑选最好测试结果。失败返回非零，并保存已完成折的记录。

## 实验配置

- configs/local.json：本地函数原样提取、保留渐进累积、MLP alpha×10 和随机 8:2 划分。源文件哈希见 configs/local_sources.json。固定参数采用论文表及本地范围/摘要；预处理移动到训练集内，不沿用测试集选最佳步骤、模拟预测和结果过滤。
- configs/grouped.json：按化学成分组合代理试件组进行 5 折评估，使用类别约束改进生成器，训练内预处理、氧化物归一化及相同 MLP 正则化。这是后续改进实验，参数预先固定，不是嵌套调参。

```powershell
python run_experiment.py --model all --config configs/grouped.json --output runs/six_grouped
```

原论文、本地不同版本与工程整理存在差异，当前版本不承诺重现发表的 R²=0.8065。所有未确认细节、兼容设置和保留行为见 RUNNING.md。使用测试曲线调整配置后，需要新的独立验证，不能仍把该曲线当作未参与选择的最终评估。

## 仅生成样本

原有生成入口仍可用，依赖仅为 NumPy 和 pandas：

```powershell
python -m pip install -r requirements.txt
python -m ndt_samples --n-virtual 100 --seed 42 --output generated/new_samples.csv
```

此入口使用改进生成器。默认目标 TC(W/mK)、类别 Saturated、排除 No.。自定义参数见 `python -m ndt_samples --help`。

## 测试与范围

```powershell
python -m unittest discover -s tests -v
```

测试覆盖生成器边界、分组隔离、训练内缩放、可复现渐进训练、真实预测指标、RF 改进路径及无效配置。虚拟数据不是独立测量，增强效果应通过真实测试验证。SHAP、全量网格搜索、所有论文图表和模型部署不属于本轮迁移范围。没有给原研究擅自添加许可证。
