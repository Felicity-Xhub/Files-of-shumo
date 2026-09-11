# A题问题4复现与模型口径

依赖 Python 3.10+、NumPy、pandas、SciPy、Matplotlib。`environment_si.csv` 与 `radius_si.csv` 分别由附件1、附件2转换为 SI 单位后得到。

```powershell
python q4_model_solve.py --environment environment_si.csv --radius-data radius_si.csv --output-dir reproduced
```

## 移动边界与守恒口径

取归一化材料坐标 \(x=r/R(t)\in[0,1]\)。假设药材在径向上均匀收缩，\(x\) 标记同一材料层；状态 \(T(x,t)\)、\(C(x,t)\) 因而随材料层演化，而非固定空间位置上的欧拉场。对圆柱径向一维传递，代码离散的连续形式为

\[
\rho(C)c_p(C)\frac{\partial T}{\partial t}
=\frac{1}{R(t)^2x}\frac{\partial}{\partial x}
\left[xk(C)\frac{\partial T}{\partial x}\right],
\]

\[
\frac{\partial C}{\partial t}
=\frac{1}{R(t)^2x}\frac{\partial}{\partial x}
\left[xD(C,T)\frac{\partial C}{\partial x}\right].
\]

中心满足零通量；表面满足 \(kT_x/R=h(T_\infty-T_s)\) 与 \(DC_x/R=h_m(C_{eq}-C_s)\)。在该材料坐标口径下，收缩速度已包含在“随材料层取时间导数”中，因此不再额外加入欧拉坐标变换项 \(x\dot R/R\)。数值离散采用圆柱有限体积与 BDF 自适应隐式积分。

## 明确假设

- 4 h 后烘房温度和含水率保持附件1末值；
- 题目未给吸附等温线，取空气含水率为有效表面平衡含水率 \(C_{eq}\)；
- 仅考虑圆柱径向传热传质，忽略端部效应；
- 不显式加入蒸发潜热耦合；
- 半径在附件2的 0--72 h 范围内线性插值，不作范围外外推。

## 输出与阈值

脚本先定位临界时刻 \(t^*\)，使 \(\max_x C(x,t^*)=0.15\) kg/kg；这不是题意“低于”的最终认证时间。随后在 60 s 输出网格上取至少晚于 \(t^*\) 120 s 的认证候选时刻，并要求 \(\max_x C<0.15-5\times10^{-5}\) kg/kg；这样认证终点在题目规定的四位小数报告中也显示为小于 0.1500。

输出包括从 0 s 开始逐 60 s 的水分分布、动态半径、表6、网格收敛、敏感性、材料坐标守恒检查、同网格固定半径基线以及图形。固定距离超过当时药材表面时，对应 CSV/Excel 单元格留空；最后一列始终为动态药材表面值。
