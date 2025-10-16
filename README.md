# DeepFunding Trial Data Analysis 🔍

Quick analysys of the DeepFunding trial dataset and a few extra experiments!

## 📦 Dataset

- 593 comparisons across 37 jurors and 45 repositories.
- 390 unique repository pairings; the median pair appears once and only seven pairs surface more than three times, led by the six match argotorg/sourcify versus consensys/teku rivalry.
- Parent track coverage stays singular and both choice labels remain present in the merged dataset.

![Comparisons per repository pair](figures/comparisons-per-repo-pair.png)

## 🧑‍⚖️ Juror Participation

- Workload remains concentrated.
  - Top five jurors contribute 25% of all votes, led by `L1Juror13` at 32 comparisons,
  - Five jurors log five or fewer (`L1Juror16` makes only three).

![Comparisons per juror](figures/comparisons-per-juror.png)

![Juror workload distribution](figures/comparisons-per-juror-hist.png)

## 🧬 Repository Coverage

- Core execution clients dominate: `erigontech/erigon` enters 51 comparisons, `ethereum/go-ethereum` 48, and both `paradigmxyz/reth` and `prysmaticlabs/prysm` 44 each.
- Despite breadth across 45 repositories, the top five represent 19% of all appearances, keeping the long tail thinly measured.

![Repository appearances](figures/appearances-per-repository.png)

## 🛡️ Robustness Checks

- Removing any single juror shifts weights modestly except for `L1Juror38`, whose absence lifts `ethereum/eips` by +0.141 (L1 redistribution 0.364).
- Pair-level sensitivity stays anchored on `ethereum/eips`; the `ethereum/eips` ↔ `ethereum/go-ethereum` matchup drives a +0.093 max delta.

![Leave-one-out juror impacts](figures/loo-juror-impacts.png)

![Leave-one-out pair impacts](figures/loo-pair-impacts.png)

![Juror redistribution](figures/loo-juror-redistribution.png)

![Repository redistribution](figures/loo-repository-redistribution.png)

## 🚨 Data Quality Signals

- Triangle checks find no order cycles (0 of 9 triangles)
- There are though six intensity loops (66.7%) with inconsistencies (>25% relative error).
  - `L1Juror9`
    - prysmaticlabs/prysm>paradigmxyz/reth (x2.00, n=1) and paradigmxyz/reth>chainsafe/lodestar (x8.00, n=1) imply prysmaticlabs/prysm>chainsafe/lodestar should be x16.00, but recorded x3.00 (n=1) → error 81.2%
  - `L1Juror6`
    - nomicfoundation/hardhat>safe-global/safe-smart-account (x8.00, n=1) and safe-global/safe-smart-account>openzeppelin/openzeppelin-contracts (x3.00, n=1) imply nomicfoundation/hardhat>openzeppelin/openzeppelin-contracts should be x24.00, but recorded x5.00 (n=1) → error 79.2%
- Average absolute side bias sits at 13.2%; L1Juror16 picks "left" 100% of the time (3/3), and four jurors display a 30% skew.

## ⚖️ Weights

### 🪪 Default Method

- [Vitalik's proposed least-squares method](https://github.com/deepfunding/scoring) baseline weights remain top heavy with `ethereum/eips` at 0.216, followed by `ethereum/go-ethereum` (0.128), `argotorg/solidity` (0.097), `foundry-rs/foundry` (0.050), and `sigp/lighthouse` (0.045).
- The top five repositories concentrate 53.5% of total weight; the top ten push that share to 71.3%, leaving the median weight at 0.0093.

![Baseline repository weights](figures/baseline-weights.png)

## 📊 Alternative Weight Methods

I've implemented a few alternative weight distribution methods. Each one produces a different allocation.

![Weight comparison across methods](figures/repo-weights-by-method.png)

- Huber-log continues to lead with Brier 0.0748, LogLoss 0.589, Accuracy 0.767, Skill +0.410, and minimal total variation drift (0.035).
- Least-squares remains a close second (Brier 0.0755, LogLoss 0.592, Accuracy 0.759, Skill +0.405) with no mass reallocation (total variation 0.000).
- Bradley-Terry regularised records the highest accuracy at 0.789 but carries total variation 0.323; the intensity-aware flavour stays brittle (Brier 0.1434, LogLoss 1.174, Skill -0.131, total variation 0.679).
- Elo (Brier 0.1018, LogLoss 0.651, Skill +0.198) and PageRank (Brier 0.1027, LogLoss 0.657, Skill +0.190) remain above the naive baseline yet lag the logistic models.


## 🤖 Models



- Jury baseline (least-squares weights) matches the default formula: Brier 0.0755, LogLoss 0.592, LogOddsRMSE 2.19, Accuracy 0.759, Skill +0.405, and zero total variation drift, giving the calibration target the other models try to beat.
- Composite ensemble stays closest to the ground-truth signal: Brier 0.0769, LogLoss 0.588, Skill +0.394, Accuracy 0.769, and the shallowest total variation shift (0.229), meaning it reallocates the least weight while still beating the baseline by ~39%.
- Arbitron trails but remains serviceable with Brier 0.0881, LogLoss 0.625, Skill +0.306, Accuracy 0.747, and a slightly higher weight drift (TV 0.261), signalling decent calibration at the cost of modest re-weighting.
- Final Seer underperforms: Brier 0.1109, LogLoss 1.548, LogOddsRMSE 6.94, Accuracy 0.727, and TV 0.226 reveal broad coverage yet substantial miscalibration, so it is not competitive without further tuning.
