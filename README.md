# MultimodalUAVOODNet

固定 8-slot、RF 主导识别、Radar 物理轨迹定位的 OOD 无人机 ENU 工程。训练数据严格只来自 `train/index.csv`，公开测试严格只来自 `test_public/index.csv`；`val/` 不会读取。

## DINOv3 ViT-S+/16 distilled EO 依赖

EO 是辅助模态，不是检测器，也不需要任何框/掩码标注。项目固定使用官方 DINOv3 ViT-S+/16 distilled 的 LVD-1689M 权重；为保证 checkpoint parity，训练和推理都只接受本地仓库与本地权重，绝不在运行期联网下载或静默回退到随机骨干。加载时只导入 `dinov3.hub.backbones`，不会通过 `hubconf.py` 连带导入不需要的分割/检测依赖。

运行环境必须是 **Python 3.10+**、PyTorch 2.0+。Python 3.9 无法导入当前官方 DINOv3 源码。

服务器上建议固定到以下位置：

```bash
git clone https://github.com/facebookresearch/dinov3.git /data1/whd/AI_wireless/dinov3-main
mkdir -p /data1/whd/AI_wireless/dinov3-main/weights
```

先在 Meta 官方 DINOv3 权重申请页接受许可；获批邮件会给出 `ViT-S+/16 distilled LVD-1689M` 的下载 URL。用邮件中的 URL 下载，文件必须放为：

```text
/data1/whd/AI_wireless/dinov3-main/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
```

例如（将 `<APPROVED_URL>` 原样替换为邮件里的 URL）：

```bash
wget -O /data1/whd/AI_wireless/dinov3-main/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth '<APPROVED_URL>'
sha256sum /data1/whd/AI_wireless/dinov3-main/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
```

校验结果必须以 `4057cbaa` 开头；工程会在任何训练前再次校验。官方 DINOv3 仓库与其 PyTorch-Hub 用法见 [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)；DINOv3 完整训练/评估环境要求 PyTorch 2.7.1+，本工程至少需要能运行该官方 backbone 的 PyTorch 版本。

## 唯一主命令

服务器完整 5-fold：

```bash
python -u train_end2end.py \
  --data-root "/data1/whd/AI_wireless/dataset" \
  --output-dir "./outputs" --folds 5 --epochs 120 \
  --dinov3-repo-dir "/data1/whd/AI_wireless/dinov3-main" \
 --eo-pretrained-path "/data1/whd/AI_wireless/dinov3-main/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
```

任意 `--folds >=2` 都可使用。若只训练一个指定折、但仍要直接生成单 checkpoint submission，可使用：

```bash
python -u train_end2end.py \
  --data-root "/data1/whd/AI_wireless/dataset" \
  --output-dir "./outputs" --folds 5 --epochs 15 --only-fold 0 \
  --single-fold-submission --reuse-folds
```

这里的 `1/5` 指的是先建立 5-fold session-CV 切分，再只使用第 0 折（4/5 训练、1/5 验证）训练/推理；它不是 5 个 checkpoint ensemble。生成前仍会强制该折 OOF parity，通过后写出 `outputs/submission.jsonl`。

先验证一个严格 session-CV fold（不生成 submission）：

```bash
python -u train_end2end.py \
  --data-root "/data1/whd/AI_wireless/dataset" \
  --output-dir "./outputs" --folds 5 --epochs 15 --only-fold 0 \
  --dinov3-repo-dir "/data1/whd/AI_wireless/dinov3-main" \
  --eo-pretrained-path "/data1/whd/AI_wireless/dinov3-main/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
```

本地完整 5-fold：

```powershell
python -u train_end2end.py `
  --data-root "E:\self_data\competition\AI+无线电\复\数据" `
  --output-dir "E:\self_data\competition\AI+无线电\复\code\uav_emu_ood\outputs" `
  --folds 5 --epochs 120 `
  --dinov3-repo-dir "E:\self_data\competition\AI+无线电\复\third_party\dinov3" `
  --eo-pretrained-path "E:\self_data\competition\AI+无线电\复\weights\dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
```

启动顺序固定为：权限/路径检查、index 驱动的数据审计、trajectory/session CV、可信度诊断、预训练 checkpoint parity probe、训练与 OOF、saved-checkpoint parity gate、checkpoint ensemble（或显式单折）、allowlist hard mask、submission 审计。任何 parity 不一致都会抛错并禁止生成 submission。

单独复核已保存 fold：

```bash
python -u parity_check.py --data-root <DATA_ROOT> --checkpoint outputs/fold_0/best.pt --oof outputs/fold_0/oof_best.npz
```
