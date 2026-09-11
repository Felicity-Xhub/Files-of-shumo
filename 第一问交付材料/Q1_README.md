# A题问题1复现说明

## 依赖

- Python 3.10+
- numpy
- pandas
- scipy
- matplotlib

## 运行模型

先运行阶段三数据审查，再在项目目录执行：

```powershell
python q1_model_solve.py --environment outputs/stage3/environment_si.csv --output-dir outputs/stage4_q1
```

默认生产网格步长为0.0015625 cm，共1281个径向节点；BDF相对、绝对误差容限分别为 `1e-10`、`1e-12`，最大内部步长10 s。脚本同时运行四级空间网格收敛检查。

## 生成Excel

`work/q1_artifact/build_result1.mjs` 使用原始 `result1.xlsx` 模板、模型生成的两个完整CSV，导出最终Excel。模型求解代码不包含用户设备绝对路径；所有输入和输出均通过命令行参数或项目相对路径配置。
