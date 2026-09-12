# A题问题3复现说明

## 依赖

- Python 3.10+
- numpy
- pandas
- scipy
- matplotlib

## 运行模型

在本目录执行：

```powershell
python q3_model_solve.py --environment environment_si.csv --output-dir reproduced
```

默认生产网格步长为 0.00078125 cm，共 2561 个径向节点；采用 BDF 自适应隐式积分，`rtol=1e-9`、`atol=1e-11`，最大内部步长 60 s。代码以 `max(C)-0.15` 的首次向下过零定位理论临界时间。

完整运行会重复网格收敛和参数敏感性分析，耗时和内存占用较高。若只复现已经审查通过的主结果，可使用：

```powershell
python q3_model_solve.py --environment environment_si.csv --output-dir reproduced --skip-diagnostics
```

## 结果口径

- 理论临界时间：57.172706 h；
- 60 s 输出网格上的认证时间：57.25 h；
- `result3.xlsx`：60–206100 s，每 60 s；0–2 cm，每 0.1 cm；水分保留四位小数。

程序先定位数值临界事件并保存事件状态，再关闭终止事件续积分至候选报告时间。206100 s 处全节点最大未舍入含水率为 0.149918566786，小于 0.14995，因此该时刻通过四位小数报告条件。采样函数禁止对真实积分区间外的时间求值；移动域调用时，域外半径输出为空值。

Excel 已随交付材料提供。模型代码输出未舍入 CSV、JSON 和 PNG；正式 Excel 由题目原始模板和同一份未舍入水分 CSV 生成，底层单元格保留未舍入值，仅显示四位小数，不手工修改最后一行。

