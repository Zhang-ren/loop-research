# 合成传感器阈值实验

`evaluate.py` 读取冻结的样本、当前划分和 `threshold` 参数。预测使用读数是否大于等于阈值，指标为准确率。该项目只使用 Python 3.12 标准库，模板通过 `python3.12` 调用解释器。

`samples.json` 包含 12 个小样本。`splits.json` 列出互不重叠的 4 个训练 ID、4 个验证 ID 和 4 个独立测试 ID。此规则基线没有拟合步骤，候选通过比较验证集上的准确率来选择。`experiment.template.json` 声明阈值 7 作为基线，允许 3、5、7，最多执行 3 个探索候选。

本地命令执行时，控制器固定 Git 提交并传入当前划分。训练/评测入口读取 `LOOP_PARAMETERS`、`LOOP_SPLIT_MANIFEST` 和 `LOOP_SPLIT`，向 `LOOP_METRICS` 写出指标和预测。数据是合成的，用于检验接入、独立复核和复跑。
