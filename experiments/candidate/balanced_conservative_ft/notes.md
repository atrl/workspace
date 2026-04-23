# balanced_conservative_ft — Conservative PPO Fine-tuning from BC Pretrain

## Hypothesis

PPO training destroys the initial policy (iter 0 is often best). By:
1. Pretraining with Behavioral Cloning (BC) on rulebase demonstrations
2. Fine-tuning with very conservative PPO (lr=1e-5, max_iters=20, high entropy)

We can get a policy that improves upon BC without the usual training degradation.

## Baseline

- BC pretrain: loss=1.6576, accuracy=0.8906 on 10K rulebase samples
- Balanced canonical: proxy ~470% / Sharpe ~0.9 / MaxDD ~50%

## Configuration Delta

| Parameter | Balanced | Conservative FT |
|-----------|----------|----------------|
| lr | 0.0003 | 0.00001 |
| max_iters | 80 | 20 |
| ent_coef | 0.01 | 0.05 |
| checkpoint_path | "" | BC pretrain actor |

## Acceptance Criteria

- Training does NOT degrade from iter 0
- Valid score improves or stays stable across iters
- Proxy return > 400% AND MaxDD < 55%

## External Reference

- Kumar et al. (2020) Conservative Q-Learning
- Nair et al. (2020) AWAC
