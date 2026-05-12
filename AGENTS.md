# IrodoriTTS + Character Reference

IrodoriTTS の基本機能 (README.md 参照) に加えて、画像を参照して生成する機能を実装する。

## Character Reference

現状、Voice Cloning (音声参照) やVoice Design (テキストキャプション) という機能で、テキストによって声のスタイルを指定することができるが、これと似たような形で、画像を参照として声のスタイルを決定する機能を実装。

# python

python3 は使用できない。
uv を使う。フォーマッタは ruff を使う。


## Server

FastAPI + OpenAPI + Scalar を使う。Scalar は以下参照:

https://scalar.com/products/api-references/integrations/fastapi.md