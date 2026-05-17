# IrodoriTTS + Speaker Inversion

音声参照のゼロショットの仕組みをベースにして、参照する音声埋め込みのみを学習する機能を作成する。

イメージとしては Textual Inversion に近い。特定の長さ (16 など) のトークンのみを学習し、他はフリーズする。
Voice Cloning は以下のような流れだが、

1. 音声VAE でエンコード
2. Speaker Encoder に通す → 声のスタイル埋め込みを得る
3. Joint Attention などに通す

今回の Speaker Inversion では 1, 2 をスキップして声のスタイル埋め込みを得て、それを同様に Joint Attention や Duration Predictor に渡すことになる。

uncond 条件は、attention mask によって attend しない方式と、ノイズ埋め込みを uncond として扱う方式の二つを用意して、選べるようにする。

## 学習

学習は通常の学習と同様、音声--テキストのペアを用いる。

### 検証

speechbrain の ECAPA-TDNN を用いた、話者類似度を指標にする。

### preview

学習中のプレビューでは学習した埋め込みを用いる。

### モデル

学習するのは声埋め込みなので、声埋め込みのみを保存する。
推論時は声埋め込みファイルを指定して音声参照をできるようにする。

# python

python3 は使用できない。
uv を使う。フォーマッタは ruff を使う。
