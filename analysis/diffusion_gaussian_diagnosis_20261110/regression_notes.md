# 回归验收记录

新诊断测试4项全部通过，包括UE ID路由、副本隔离、逐用户损失分解、条件采样RNG隔离、完整快照反事实、几何预测跳数与真实环境step一致（两类actor）。

扩展回归：`tests/test_diffusion_gaussian_diagnosis.py tests/test_hop_service_training.py tests/test_diffusion_dppo.py` 共32通过、1失败。失败为原有`test_numpy_numba_and_terminal_history`最后的异常类型期望：终止后重复提交上一时刻动作时，旧环境prepare先校验候选ID，候选变化会先抛ValueError，而测试期望RuntimeError("ended")。此前整个回合的NumPy/Numba物理状态、奖励和真正终止断言均通过。production environment.py与HEAD无差异，该错误优先级不是新诊断引入；本轮未修改原训练执行路径以掩盖此项失败。

该问题发生在已经终止后再次提交过期动作的非法调用，实际本轮采集与评估均在400步终止，无额外step。它不解释正常训练中覆盖下降。后续可单独修复终止检查优先级及测试固定随机种子，但不将其算作本次科学诊断已修复的问题。
