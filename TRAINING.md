# Irodori-TTS 学習ガイド

このドキュメントでは、Irodori-TTS のデータセット形式と学習設定について詳しく説明します。

---

## 目次

1. [全体的なワークフロー](#全体的なワークフロー)
2. [データセット形式](#データセット形式)
3. [マニフェストの作成](#マニフェストの作成prepare_manifestpy)
4. [学習の実行](#学習の実行trainpy)
5. [設定ファイル (YAML)](#設定ファイル-yaml)
6. [LoRA ファインチューニング](#lora-ファインチューニング)
7. [VoiceDesign（キャプション条件付き）学習](#voicedesignキャプション条件付き学習)
8. [マルチ GPU 学習](#マルチ-gpu-学習)

---

## 全体的なワークフロー

```
HuggingFace Dataset (音声 + テキスト)
        ↓
prepare_manifest.py  ←  音声を DACVAE でエンコード → .pt ファイル
        ↓
train_manifest.jsonl + latents/*.pt
        ↓
train.py  ←  configs/*.yaml
        ↓
checkpoint .pt / .safetensors
```

---

## データセット形式

### JSONL マニフェスト

学習データは **JSONL 形式**のマニフェストファイルで管理します。1 行 = 1 サンプルです。

#### フィールド一覧

| フィールド | 型 | 必須 | 説明 |
|---|---|---|---|
| `text` | string | ✅ | 学習テキスト（日本語） |
| `latent_path` | string | ✅ | DACVAE latent `.pt` ファイルへの相対パス |
| `num_frames` | integer | ✅ | latent のフレーム数 |
| `speaker_id` | string | — | 話者 ID（例: `myorg/dataset:speaker_001`） |
| `caption` | string | — | スタイル制御キャプション（VoiceDesign 用） |

#### サンプル

```jsonl
{"text": "こんにちは、今日はいい天気ですね。", "latent_path": "data/latents/00000000.pt", "num_frames": 420, "speaker_id": "myorg/my_dataset:speaker_001"}
{"text": "音声合成モデルの学習を行います。", "latent_path": "data/latents/00000001.pt", "num_frames": 510, "speaker_id": "myorg/my_dataset:speaker_002", "caption": "落ち着いた、近い距離感の女性話者"}
```

### ディレクトリ構成

```
data/
├── train_manifest.jsonl          # メインのマニフェスト
├── train_manifest.rank00.jsonl   # マルチ GPU 時：ランクごとのシャード
├── train_manifest.rank01.jsonl
└── latents/
    ├── 00000000_00000000.pt      # DACVAE latent（形状: T × 32）
    ├── 00000001_00000001.pt
    └── ...
```

### Latent ファイルの仕様

| 項目 | 内容 |
|---|---|
| フォーマット | PyTorch テンソル (`.pt`) |
| 形状 | `(T, 32)`（T: フレーム数、32: 潜在次元） |
| コーデック | [Aratako/Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim) |
| ラウドネス正規化 | デフォルト -16 dB |

---

## マニフェストの作成（`prepare_manifest.py`）

HuggingFace データセットから音声をエンコードし、JSONL マニフェストと latent ファイルを生成します。

### 基本的な使い方

```bash
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --speaker-column speaker \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

### 主なオプション

#### データセット入力

| オプション | デフォルト | 説明 |
|---|---|---|
| `--dataset` | — | HuggingFace データセット ID（必須） |
| `--config` | — | データセットのサブセット名 |
| `--split` | `train` | 使用するスプリット |
| `--data-files` | — | ローカルファイルのパス（例: `train=data/train.jsonl`） |
| `--streaming` | False | ストリーミングモードで読み込む |
| `--trust-remote-code` | False | リモートコードの実行を許可 |

#### カラム指定

| オプション | デフォルト | 説明 |
|---|---|---|
| `--audio-column` | — | 音声カラム名（必須） |
| `--text-column` | — | テキストカラム名（必須） |
| `--speaker-column` | — | 話者 ID カラム名（複数指定可） |
| `--caption-column` | — | キャプションカラム名（VoiceDesign 用） |
| `--text-normalize` | True | 日本語テキストの正規化 |
| `--speaker-id-prefix` | データセット名 | 話者 ID のプレフィックス |

#### 音声処理

| オプション | デフォルト | 説明 |
|---|---|---|
| `--codec-repo` | `Aratako/Semantic-DACVAE-Japanese-32dim` | 使用するコーデック |
| `--normalize-db` | `-16.0` | ラウドネス正規化の目標値（dB）。`none` で無効化 |
| `--target-sample-rate` | — | リサンプリング先のサンプルレート |
| `--min-sample-rate` | `0`（無効） | これ未満のサンプルレートをスキップ |
| `--max-seconds` | — | 音声の最大長（秒）。超えた場合はトリム |

#### マルチ GPU・パフォーマンス

| オプション | デフォルト | 説明 |
|---|---|---|
| `--num-gpus` | — | 使用 GPU 数（複数 GPU でエンコードを並列化） |
| `--merge-output` | False | マルチ GPU 後にシャードを結合 |
| `--prefetch` | `0` | プリフェッチキューサイズ |
| `--prefetch-workers` | `1` | プリフェッチワーカー数 |
| `--max-samples` | — | 書き出すサンプル数の上限 |

### マルチ GPU での実行例

```bash
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --speaker-column speaker \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --num-gpus 4 \
  --merge-output
```

### VoiceDesign 用（キャプション付き）

```bash
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --caption-column caption \
  --speaker-column speaker \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

---

## 学習の実行（`train.py`）

### 基本的な使い方

```bash
# シングル GPU
uv run python train.py \
  --config configs/train_500m_v2.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts

# マルチ GPU (DDP)
uv run torchrun --nproc_per_node 4 train.py \
  --config configs/train_500m_v2.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts
```

### チェックポイントの再開・初期化

```bash
# 学習を途中から再開（オプティマイザの状態も含む）
uv run python train.py \
  --config configs/train_500m_v2.yaml \
  --manifest data/train_manifest.jsonl \
  --resume outputs/irodori_tts/checkpoint_0010000.pt

# リリース済みモデルから重みのみ初期化（オプティマイザはリセット）
uv run python train.py \
  --config configs/train_500m_v2_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --init-checkpoint released_model.safetensors \
  --output-dir outputs/irodori_tts_lora
```

### 主なオプション

#### 基本設定

| オプション | デフォルト | 説明 |
|---|---|---|
| `--config` | — | YAML 設定ファイルのパス |
| `--manifest` | — | JSONL マニフェストのパス（必須） |
| `--output-dir` | `outputs/irodori_tts` | チェックポイントの保存先 |
| `--device` | `cuda` | 使用デバイス（`cuda` / `mps` / `cpu`） |
| `--seed` | `0` | 乱数シード |
| `--resume` | — | チェックポイントから再開 |
| `--init-checkpoint` | — | 重みのみ初期化（`.pt` / `.safetensors`） |

#### バッチ・データ

| オプション | デフォルト | 説明 |
|---|---|---|
| `--batch-size` | `8` | バッチサイズ |
| `--gradient-accumulation-steps` | `1` | 勾配累積ステップ数 |
| `--num-workers` | `2` | DataLoader のワーカー数 |
| `--max-text-len` | `256` | テキストトークンの最大長 |
| `--max-caption-len` | `max-text-len` と同値 | キャプショントークンの最大長 |

#### 最適化

| オプション | デフォルト | 説明 |
|---|---|---|
| `--optimizer` | — | `adamw` または `muon` |
| `--lr` | `1e-4` | 学習率 |
| `--weight-decay` | `0.01` | 重み減衰 |
| `--lr-scheduler` | `none` | `none` / `cosine` / `wsd`（Warmup-Stable-Decay） |
| `--warmup-steps` | `0` | ウォームアップステップ数 |
| `--stable-steps` | `0` | wsd スケジューラの安定期ステップ数 |
| `--min-lr-scale` | `0.1` | 学習率の最小スケール |
| `--max-steps` | `200000` | 最大学習ステップ数 |
| `--grad-clip-norm` | `1.0` | 勾配クリッピングのノルム |

#### 精度・高速化

| オプション | デフォルト | 説明 |
|---|---|---|
| `--precision` | `bf16` | 計算精度（`fp32` / `bf16`） |
| `--tf32` | False | TF32 matmul 高速化（A100 など） |
| `--compile-model` | False | `torch.compile` を有効化 |

#### Dropout（条件付けのドロップアウト）

| オプション | デフォルト | 説明 |
|---|---|---|
| `--text-condition-dropout` | `0.1` | テキスト条件付けのドロップアウト率 |
| `--speaker-condition-dropout` | `0.1` | 話者条件付けのドロップアウト率 |
| `--caption-condition-dropout` | `0.1` | キャプション条件付けのドロップアウト率 |

#### チェックポイント・ロギング

| オプション | デフォルト | 説明 |
|---|---|---|
| `--save-every` | `1000` | チェックポイントの保存間隔（ステップ） |
| `--log-every` | `100` | ログの出力間隔（ステップ） |
| `--checkpoint-best-n` | `0`（全保存） | 検証スコアが良い上位 N 件のみ保持 |
| `--valid-ratio` | `0.0` | 検証データの割合 |
| `--valid-every` | `0` | 検証の実施間隔（ステップ） |
| `--wandb` | False | Weights & Biases ロギングを有効化 |
| `--wandb-project` | `Irodori-TTS` | W&B プロジェクト名 |

---

## 設定ファイル (YAML)

コマンドライン引数の代わりに YAML ファイルで設定をまとめて管理できます。

### 設定ファイル一覧

| ファイル | 説明 |
|---|---|
| [configs/train_500m_v2.yaml](configs/train_500m_v2.yaml) | 500M v2 ベースモデル（話者条件付き） |
| [configs/train_500m_v2_lora.yaml](configs/train_500m_v2_lora.yaml) | 500M v2 LoRA ファインチューニング |
| [configs/train_500m_v2_voice_design.yaml](configs/train_500m_v2_voice_design.yaml) | 500M v2 VoiceDesign（キャプション条件付き） |
| [configs/train_500m_v2_voice_design_lora.yaml](configs/train_500m_v2_voice_design_lora.yaml) | 500M v2 VoiceDesign LoRA ファインチューニング |

### YAML の構造

```yaml
model:
  latent_dim: 32
  model_dim: 1280
  num_layers: 12
  num_heads: 20
  # ... ModelConfig の全フィールドを記述可能

train:
  batch_size: 80
  learning_rate: 1e-4
  optimizer: muon
  lr_scheduler: wsd
  warmup_steps: 1000
  stable_steps: 44000
  max_steps: 50000
  precision: bf16
  allow_tf32: true
  # ... TrainConfig の全フィールドを記述可能
```

### 主なモデル設定パラメータ（`ModelConfig`）

| パラメータ | v2 500M | 2.5B | 説明 |
|---|---|---|---|
| `latent_dim` | `32` | `128` | Latent の次元数 |
| `model_dim` | `1280` | `2048` | モデルの埋め込み次元 |
| `num_layers` | `12` | `24` | Diffusion Transformer の層数 |
| `num_heads` | `20` | `16` | アテンションヘッド数 |
| `text_dim` | `512` | `1280` | テキストエンコーダの次元 |
| `text_layers` | `10` | `14` | テキストエンコーダの層数 |
| `speaker_dim` | `768` | `1280` | 話者エンコーダの次元 |
| `speaker_layers` | `8` | `14` | 話者エンコーダの層数 |
| `use_caption_condition` | `false` | `false` | キャプション条件付けを有効化 |
| `adaln_rank` | `192` | `256` | AdaLN の低ランク次元 |

---

## LoRA ファインチューニング

リリース済みモデルを少ないデータで効率よくファインチューニングするための手法です。

### 実行例

```bash
uv run python train.py \
  --config configs/train_500m_v2_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --init-checkpoint released_model.safetensors \
  --output-dir outputs/irodori_tts_lora
```

### LoRA 設定パラメータ

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `--lora` | False | LoRA を有効化 |
| `--lora-r` | `16` | LoRA のランク |
| `--lora-alpha` | `32` | LoRA のスケーリング係数 |
| `--lora-dropout` | `0.0` | LoRA のドロップアウト率 |
| `--lora-bias` | `none` | バイアスの学習対象（`none` / `all` / `lora_only`） |
| `--lora-target-modules` | — | 対象モジュールのプリセットまたはカスタムリスト |

### `--lora-target-modules` プリセット一覧

| プリセット | 対象モジュール |
|---|---|
| `text_attn_mlp` | テキストエンコーダのアテンション + MLP |
| `caption_attn_mlp` | キャプションエンコーダのアテンション + MLP |
| `speaker_attn_mlp` | 話者エンコーダのアテンション + MLP |
| `diffusion_attn` | Diffusion Transformer のアテンション層 |
| `diffusion_attn_mlp` | Diffusion のアテンション + MLP |
| `all_attn` | 全アテンションブロック |
| `diffusion_full` | Diffusion スタック全体（広範囲） |
| `adaln` | AdaLN 層のみ |
| `conditioning` | 条件付け側のプロジェクション |
| `all_attn_mlp` | 全アテンション + MLP ブロック |
| `all_linear` | 全 `nn.Linear` 層 |

---

## VoiceDesign（キャプション条件付き）学習

テキストキャプションで音声スタイルを制御するモデルを学習します。

### マニフェスト準備

`caption` フィールドに話者スタイルの自然言語説明を追加します。

```jsonl
{"text": "こんにちは。", "caption": "落ち着いた、低めの声の男性話者", "latent_path": "data/latents/00000000.pt", "num_frames": 300, "speaker_id": "myorg/dataset:spk001"}
```

### 学習実行

```bash
uv run python train.py \
  --config configs/train_500m_v2_voice_design.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_voice_design
```

### VoiceDesign 固有の設定

| パラメータ | 説明 |
|---|---|
| `use_caption_condition: true` | キャプション条件付けを有効化 |
| `caption_warmup` | キャプションなしのウォームアップを行うか |
| `caption_warmup_steps` | キャプションウォームアップのステップ数 |
| `caption_condition_dropout` | キャプション条件付けのドロップアウト率 |

---

## マルチ GPU 学習

### DDP（DistributedDataParallel）

```bash
uv run torchrun --nproc_per_node 4 train.py \
  --config configs/train_500m_v2.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts
```

マルチ GPU 学習時は、マニフェストを事前にランクごとにシャーディングしておくと効率的です。

```
data/train_manifest.rank00.jsonl  # GPU 0 用
data/train_manifest.rank01.jsonl  # GPU 1 用
data/train_manifest.rank02.jsonl  # GPU 2 用
data/train_manifest.rank03.jsonl  # GPU 3 用
```

`--manifest data/train_manifest.jsonl` と指定するだけで、`.rank{N}.jsonl` ファイルが存在する場合は自動的にそちらが使用されます。

### マニフェストのシャーディング

```bash
# マルチ GPU でエンコードし、シャードを自動生成・結合
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --num-gpus 4 \
  --merge-output       # 結合済みマニフェストも生成
  # --keep-shards      # シャードファイルも保持する場合
```
