# UAV ENU OOD — 多模态 Set Prediction

本工程只读取 `train/index.csv` 的 7362 条训练样本与
`test_public/index.csv` 的 487 条测试样本，不扫描目录混入旧文件。
`val/` 不参与训练、验证或模型选择。

## 当前架构

```text
4-node IQ (Raw-IQ CNN + sr-aware STFT CNN) ──> RF tokens
Radar [E,N,U,rel_time_s] ────────────────────> Point Transformer tokens
optional EO image (letterbox + DINOv3 patch tokens) ─> EO tokens
                                                     │
3 learnable unordered Target Queries
  self-attention -> Radar cross-attention -> RF cross-attention -> EO cross-attention
                                                     │
  objectness, 8-way model logits, ENU mean, ENU log-sigma
```

- 三个 query 不绑定型号或坐标；训练使用 Hungarian matching。
- Radar 没有手写 yaw/translation 校准、匀速外推、轨迹锚点、手工统计特征或固定型号槽位。
- ENU 和 Radar 标准化严格由当前 train fold 统计；目标位置直接端到端回归。
- `allowlist` 仅在最终 model logits 上 hard-mask，再生成集合，提交审计强制验证违规数为零。
- 训练、验证、parity、测试完全复用同一 `Dataset` 与 `run_inference` 路径。旧固定-slot checkpoint 会被拒绝。

每个 epoch 输出并保存：micro/macro F1、exact-set accuracy、count accuracy、E/N/U MAE、3D mean/median/P75/P90/P95，以及按目标数量、model ID 的定位误差。工程不会把任何自定义指标称为 LocMass。

## 服务器：完整五折训练与提交

```bash
python -u train_end2end.py \
  --data-root "/data1/whd/AI_wireless/dataset" \
  --output-dir "./outputs_set_query" \
  --folds 5 --epochs 120 \
  --model-scale auto \
  --dinov3-repo-dir "/data1/whd/AI_wireless/dinov3-main" \
  --eo-pretrained-path "/data1/whd/AI_wireless/dinov3-main/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
```

输出为 `outputs_set_query/submission.jsonl`。每个 fold 的 best checkpoint 都先通过 validation-as-test parity；任何 parity 失败会直接中止，禁止生成 submission。

## 先跑 fold 0（不提交）

```bash
python -u train_end2end.py \
  --data-root "/data1/whd/AI_wireless/dataset" \
  --output-dir "./outputs_set_query_fold0" \
  --folds 5 --epochs 15 --only-fold 0 \
  --model-scale auto \
  --dinov3-repo-dir "/data1/whd/AI_wireless/dinov3-main" \
  --eo-pretrained-path "/data1/whd/AI_wireless/dinov3-main/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
```

若需以该单折 checkpoint 直接生成调试提交，增加 `--single-fold-submission`。它不是五折 ensemble。

## EO 消融

EO 是可选模态。没有可用本地 DINO 权重、或严格 CV 未验证其收益时，直接加：

```bash
--disable-eo
```

此模式不会导入或下载 DINO，网络使用显式 missing-modality token。启用 EO 时要求 Python 3.10+，DINOv3 仓库和权重都必须在本地；工程不会联网下载权重。

## 单独复核 parity

```bash
python -u parity_check.py \
  --data-root "/data1/whd/AI_wireless/dataset" \
  --checkpoint "./outputs_set_query/fold_0/best.pt" \
  --oof "./outputs_set_query/fold_0/oof_best.npz"
```
