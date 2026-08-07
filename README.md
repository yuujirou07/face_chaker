# python_proj

MediaPipeとOpenCVでWebカメラの手・顔ランドマークを取得するPythonプロジェクトです。

`face_checker.py` は、顔の5つの基準点から作る比率を100フレーム分平均して登録し、登録済みの顔か未登録の顔かを判定します。

## 必要環境

- Python 3.12
- Webカメラ
- `mediapipe`
- `opencv-python`
- `numpy`
- `evdev`

仮想環境を作成する場合は次を実行します。

```bash
python -m venv .venv
.venv/bin/python -m pip install mediapipe opencv-python numpy evdev
```

以下のモデルファイルがプロジェクト直下に必要です。

- `face_landmarker.task`
- `hand_landmarker.task`

## 顔の登録と照合

```bash
.venv/bin/python face_checker.py
```

1. `1` を入力すると顔を登録します。
2. 名前を入力して確認すると、顔の特徴を100フレーム採取します。
3. `<名前>_face_landmarks.json` に平均値を保存します。
4. `2` を入力すると登録済みJSONと比較します。

照合画面には次のどちらかが表示されます。

- `Recognized: <名前>`: 最も近い登録データの誤差がしきい値以下
- `Unknown face`: 登録済みの誰とも一致しない顔

`Esc` で終了します。

## 保存する特徴量

次の5点を使います。

- 左右の目尻: `446`, `226`
- 上唇中央: `0`
- 鼻上部: `1`
- 両目の中間: `168`

各点の組み合わせ10通りの距離を求め、左右の目尻間距離で割った比率として保存します。顔の表示サイズやカメラとの距離による影響を減らすためです。

`MATCH_THRESHOLD` は `face_checker.py` の照合しきい値です。誤って他人を本人と判定する場合は小さく、本人が未登録になる場合は大きく調整します。

## 手のランドマーク表示

```bash
.venv/bin/python main.py
```

左手のランドマークをカメラ映像へ描画します。

## 注意

`*_face_landmarks.json` は顔の特徴量を含むファイルです。共有リポジトリへ追加したり、他者へ渡したりする前に内容を確認してください。
