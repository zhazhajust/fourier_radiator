完全可以再把 $\omega$ 按

$$
\omega_u \;\equiv\; \frac{2\pi c}{\lambda_u}
$$

来归一化。这样做在自由电子激光 / 轫致辐射的文献里很常见，因为
$\bar\omega = \omega/\omega_u$ 就直接对应“谐波序号”——第一谐波 $\bar\omega=1$、第二谐波 $\bar\omega=2$，等等。
下面把前面已经“去掉 $c$”的结果再做一次缩放，就得到只含 $\lambda_u$ 而不含 $c$ 的最终表达式。

---

## 1  重新选择无量纲变量

| 物理量 | 原始量      | 第一步（乘 $c$）                                          | 第二步（除 $\lambda_u$ 或 $\omega_u$）                                                   |
| --- | -------- | --------------------------------------------------- | --------------------------------------------------------------------------------- |
| 时间  | $t$      | $\tau = c\,t$ （单位：米）                                | $s = \dfrac{\tau}{\lambda_u} = \dfrac{c\,t}{\lambda_u}$（纯无量纲）                     |
| 频率  | $\omega$ | $\bar\omega_1 = \dfrac{\omega}{c}$（单位：rad m$^{-1}$） | $\bar\omega = \dfrac{\omega}{\omega_u} = \dfrac{\omega\,\lambda_u}{2\pi c}$（纯无量纲） |

配套地，把轨迹写成

$$
\tilde{\mathbf R}(s)\;\equiv\;\frac{\mathbf r\!\left(t=\frac{\lambda_u}{c}\,s\right)}{\lambda_u},
\quad\text{于是}\quad
\boldsymbol\beta(s) = \frac{d\tilde{\mathbf R}}{ds},\;
\boldsymbol\beta'(s)=\frac{d\boldsymbol\beta}{ds}.
$$

---

## 2  换元对各个因子影响

1. **积分测度**

   $$
   dt = \frac{\lambda_u}{c}\,ds.
   $$

2. **加速度项**

   $$
   \dot{\boldsymbol\beta}(t)=\frac{d\boldsymbol\beta}{dt}
        =\frac{c}{\lambda_u}\,\boldsymbol\beta'(s).
   $$

3. **二者相乘**

   $$
   \dot{\boldsymbol\beta}\,dt
     =\frac{c}{\lambda_u}\boldsymbol\beta'\;\frac{\lambda_u}{c}ds
     =\boldsymbol\beta'\,ds,
   $$

   正好抵消——芯子里依旧没有任何 $c$ 或 $\lambda_u$。

4. **指数相位**

   $$
   i\omega\Bigl[t-\frac{\mathbf n\!\cdot\!\mathbf r(t)}{c}\Bigr]
   \;=\;i\omega_u\bar\omega\;
          \Bigl[\frac{\lambda_u}{c}s-\frac{\mathbf n\!\cdot\!\mathbf r(t)}{c}\Bigr]
   = i\,2\pi\bar\omega\Bigl[s-\mathbf n\!\cdot\!\tilde{\mathbf R}(s)\Bigr].
   $$

   仅多了一个整洁的 $2\pi$。

---

## 3  变量替换对导数的“雅可比因子”

我们真正关心的是
$\displaystyle\frac{d^2I}{d\bar\omega\,d\Omega}$。
链式法则给出

$$
\frac{d^2I}{d\bar\omega\,d\Omega}
     =\frac{d\omega}{d\bar\omega}
      \frac{d^2I}{d\omega\,d\Omega}
     =\omega_u\,\frac{d^2I}{d\omega\,d\Omega}.
$$

---

## 4  集中整理所有前因子

原来（已去掉 $c$ 之后）的前因子是
$\displaystyle \frac{e^2}{16\pi^3\varepsilon_0}$.
再乘一个 $\omega_u = 2\pi c/\lambda_u$ 变成

$$
\frac{e^2}{16\pi^3\varepsilon_0}
\;\times\;\frac{2\pi c}{\lambda_u}
\;=\;
\frac{e^2}{8\pi^2\varepsilon_0\,\lambda_u},
$$

其中 $c$ 已经与上一步抵消过，所以最终只剩 $1/\lambda_u$。

---

## 5  归一化到 $\omega_u$ 的最终公式

$$
\boxed{%
\frac{d^{2}I}{d\bar\omega\,d\Omega}
  =\frac{e^{2}}{8\pi^{2}\varepsilon_{0}\,\lambda_u}\;
   \left|
     \int_{-\infty}^{+\infty}
        e^{\,i\,2\pi\bar\omega\,[\,s-\mathbf n\!\cdot\!\tilde{\mathbf R}(s)\,]}
        \frac{\mathbf n\times\bigl[(\mathbf n-\boldsymbol\beta)\times\boldsymbol\beta'\bigr]}
             {(1-\boldsymbol\beta\!\cdot\!\mathbf n)^{2}}
     \,ds
   \right|^{2}}
$$

* 变量 **$s$** 是“走过的周期数”，$\bar\omega$ 就是“谐波号”；
* $\boldsymbol\beta(s)=d\tilde{\mathbf R}/ds$ 与 $\boldsymbol\beta'(s)=d\boldsymbol\beta/ds$ 完全无量纲；
* 指数里的 $2\pi$ 体现了“每跨过一次周期，相位多转一圈”的直观含义；
* 前因子变成 $\propto 1/\lambda_u$，没有任何 $c$。

---

### 小结

* **再归一化到 $\omega_u=2\pi c/\lambda_u$** 只是把频率刻度从“光程倒数”换成“每周期相位转数”；
* 积分核本身不受影响，唯独前因子从 $\dfrac{e^2}{16\pi^3\varepsilon_0}$ 变为 $\dfrac{e^2}{8\pi^2\varepsilon_0\lambda_u}$；
* 这样写对分析谐波结构或做数值傅里叶变换最方便，常用于 FEL 和同步辐射计算。
