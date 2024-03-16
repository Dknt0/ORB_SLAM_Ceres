Ceres 使用技巧
===

`ceres_problem.EvaluateResidualBlock` 返回的误差是 $\frac{1}{2}\bold{r}^{T}\bold{I}\bold{r} $，是 g2o 返回值的 1/2，在进行外点剔除时需要注意。

`PoseOptimization` 返回值（内点数）会影响关键帧判断。

Ceres 优化器会在某一次优化失败后一直失败，很神奇

> 已解决，没有初始化位姿

全局 BA 中如何开启边缘化?
