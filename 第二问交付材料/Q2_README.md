# A题问题2复现说明

依赖 Python 3.10+、NumPy、pandas、SciPy、Matplotlib。`environment_si.csv` 是由附件1审查后得到的 SI 单位边界数据；脚本和数据路径均通过参数配置，不含设备绝对路径。

```powershell
python q2_model_solve.py --environment environment_si.csv --output-dir reproduced
```

默认使用0.003125 cm径向网格、BDF自适应隐式积分、相对误差容限 `1e-9`、绝对误差容限 `1e-11` 和20 s最大内部时间步。脚本输出每1 s、每0.1 cm的完整结果，并执行网格收敛、守恒和参数敏感性检查。

若需从模板重建 Excel，在已配置 `@oai/artifact-tool` 的 Node.js 环境中运行：

```powershell
node build_result2.mjs <result2模板.xlsx> reproduced/q2_temperature_full.csv reproduced/q2_moisture_full.csv reproduced/result2.xlsx reproduced/previews
```
