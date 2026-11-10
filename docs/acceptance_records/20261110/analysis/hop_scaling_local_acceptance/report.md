# 跳数缩放：Diffusion DPPO 与 Gaussian MAPPO

两组均学习运动及UE评分，使用相同Top-K、速度缩放与真实位移历史。主要比较同预算last；扩散独有的示范和预训练成本须另计。
场景级bootstrap不代表训练种子置信区间；MC诊断使用各策略自身on-policy轨迹。

| 模型 | 平均覆盖 | 后100步覆盖 | 终态覆盖 | 吞吐Mbps | 回报 | 最大单步跌幅均值 |
|---|---:|---:|---:|---:|---:|---:|
| diffusion last | 41.199% | 43.673% | 41.333% | 315.145 | 181.604 | 13.333pp |
| gaussian last | 36.337% | 36.010% | 37.667% | 257.042 | 159.056 | 12.667pp |
| diffusion best | 45.674% | 46.507% | 46.667% | 355.818 | 201.674 | 15.667pp |
| gaussian best | 41.486% | 42.710% | 42.000% | 299.812 | 181.933 | 12.667pp |
| diffusion step0 | 45.674% | 46.507% | 46.667% | 355.818 | 201.674 | 15.667pp |
| gaussian step0 | 41.486% | 42.710% | 42.000% | 299.812 | 181.933 | 12.667pp |

运行成本与预训练成本：

```json
{
  "diffusion": {
    "online": {
      "elapsed_seconds": 90.47399294700153,
      "env_steps": 2048,
      "peak_rss_mib": 711.48046875
    },
    "pretraining": {
      "elapsed_seconds": 3.8589389529952314,
      "epoch": 1,
      "batches": 29,
      "peak_rss_mib": 557.9921875
    },
    "demonstrations": {
      "elapsed_seconds": 1.4491387759990175,
      "expected_joint_transitions": 800
    },
    "model": {
      "actor_parameters": 324867,
      "critic_parameters": 1677057
    }
  },
  "gaussian": {
    "online": {
      "elapsed_seconds": 34.264437439000176,
      "env_steps": 2048,
      "peak_rss_mib": 682.26953125
    },
    "model": {
      "actor_parameters": 291270,
      "critic_parameters": 1677057
    }
  }
}
```
