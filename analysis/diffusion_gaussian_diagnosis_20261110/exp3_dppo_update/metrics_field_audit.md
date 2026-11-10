# 历史日志字段审计

每条metrics.csv记录为一次rollout的所有PPO minibatch/epoch统计均值，不是逐minibatch原始数据。
KL=(ratio-1)-log(ratio)在旧策略样本上平均；扩散按每UAV、每去噪转移计算，Gaussian按每UAV联合latent计算，二者不是同一个最终动作KL。
denoise_8为第一次反向转移z8→z7，denoise_1为最后z1→z0。
**actor_grad_norm在全局裁剪前记录；运动头、评分头和编码器梯度在裁剪后记录。** 分支间比例可比较，不能拿分支范数直接除以日志的裁剪前总范数。

### diffusion

`boundary_clip_fraction`, `candidate_count_max`, `candidate_count_mean`, `multi_candidate_fraction`, `actor_loss`, `ppo_approx_kl`, `ppo_clip_fraction`, `ppo_sample_kl`, `value_loss`, `actor_grad_norm`, `critic_grad_norm`, `encoder_grad_norm`, `movement_head_grad_norm`, `score_head_grad_norm`, `critic_value_clip_fraction`, `denoise_8_kl`, `denoise_8_clip`, `denoise_7_kl`, `denoise_7_clip`, `denoise_6_kl`, `denoise_6_clip`, `denoise_5_kl`, `denoise_5_clip`, `denoise_4_kl`, `denoise_4_clip`, `denoise_3_kl`, `denoise_3_clip`, `denoise_2_kl`, `denoise_2_clip`, `denoise_1_kl`, `denoise_1_clip`, `candidates_empty_kl`, `candidates_empty_clip`, `candidates_within_capacity_kl`, `candidates_within_capacity_clip`, `candidates_over_capacity_kl`, `candidates_over_capacity_clip`, `critic_return_normalizer_mean`, `critic_return_normalizer_std`, `critic_average_normalizer_mean`, `critic_average_normalizer_std`

### gaussian

`boundary_clip_fraction`, `candidate_count_max`, `candidate_count_mean`, `multi_candidate_fraction`, `actor_loss`, `ppo_approx_kl`, `ppo_clip_fraction`, `ppo_sample_kl`, `value_loss`, `actor_grad_norm`, `critic_grad_norm`, `encoder_grad_norm`, `movement_head_grad_norm`, `score_head_grad_norm`, `critic_value_clip_fraction`, `gaussian_joint_kl`, `gaussian_joint_clip`, `candidates_empty_kl`, `candidates_empty_clip`, `candidates_within_capacity_kl`, `candidates_within_capacity_clip`, `candidates_over_capacity_kl`, `candidates_over_capacity_clip`, `critic_return_normalizer_mean`, `critic_return_normalizer_std`, `critic_average_normalizer_mean`, `critic_average_normalizer_std`

## 缺失的历史明细

- 历史逐minibatch advantage均值/方差
- 历史未裁剪ratio分位数
- 历史运动/评分log密度拆分
- 裁剪前分支梯度及共享编码器的分支贡献
- 历史同一观察在更新前后的最终动作分布

新的likelihood_audit_*.csv是冻结模型在新采样链上的密度拆分与更新前重算核查，不是补造历史优化器更新明细。
