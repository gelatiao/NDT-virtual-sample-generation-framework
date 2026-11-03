# 数据来源与字段

samples.tsv 的 66 行数据与论文补充材料首个数据表数值完全一致。正文称 Supplementary Table S1，补充文件标题为 Table 2 / Sandstone thermal conductivity data，注释为三位小数。

论文：Hu et al., Measurement 256 (2025), 118248，DOI 10.1016/j.measurement.2025.118248。

输入为速度、SiO2、Al2O3、K2O、CaO、Fe2O3、饱和状态、硬度八列，目标 TC；No. 只用于记录身份，不进入模型。原始字段名和本地脚本短名称通过 experiment.py 显式映射。硬度在论文描述为 Leeb hardness，原表字段标记为 Hardness(HRC)，保留原表标识，不擅自改换单位。

本地 data.csv 在舍入至三位小数后与公布表一致。data_replaced.csv 将编号 17、28、32 的 TC 从 2.265、2.525、2.052 改为 2.530，主程序不自动采用这些目标替换。允许 --input 显式指定处理版本，输出会记录不同输入 SHA256。

原始五个氧化物列总和不是 100%；归一化作为独立配置项执行，不改写源表。数据未提供真实试件 ID，grouped 流程仅用相同化学成分组做代理。
