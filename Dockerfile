# Ubuntuのベースイメージを使用
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# 必要なソフトウェアをインストール
RUN apt-get update && apt-get install -y python3 python3-pip

# 作業ディレクトリを設定
WORKDIR /app

# 依存関係をコピーしてインストール
COPY requirements.txt /app
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r requirements.txt

# アプリケーションのコードをコピー
COPY ./chatswitch.py /app/
COPY ./user_config_docker.json /app/user_config.json
COPY ./locale /app/locale

# アプリケーションを実行
CMD ["python3", "chatswitch.py"]
