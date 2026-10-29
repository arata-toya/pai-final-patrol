# AI壁面巡視ロボット V3

カメラ画像から**正常面・汚れ・ひび割れ候補**を分類し、候補への接近に応じて減速・停止します。停止をオドメトリで確認して画像と位置を保存し、約3秒後に巡視を再開。複数箇所の点検とHTMLレポートの自動生成までを実装しました。V1/V2を残した発展版です。

共通の青い枠は領域抽出と距離推定に使用し、分類用CNNには枠を除いた内部のグレースケール画像だけを入力します。

## 確認済みの結果（2026-10-29）

現行モデル `v3.0.2-camera-augmentation`（SHA-256先頭 `0148fd9528b8`）で、Gazeboの**4条件・6走行が6/6タスク成功**。各条件の最初の1回では4/4成功、CASE_Dは録画の再実行も含め3/3成功です。独立した多数条件の性能推定ではありません。

| コース | 配置：ワールドx座標 | 点検期待数／走行 | 成功／走行数 |
| --- | --- | --- | --- |
| CASE_A | 正常4m、汚れ7m、ひび割れ10m | 1 | 1/1 |
| CASE_B | 正常4m、別形状ひび割れ7m、汚れ10m | 1 | 1/1 |
| CASE_C | 正常4m、汚れ7m、正常10m | 0 | 1/1 |
| CASE_D | 正常4m、ひび割れ7m、別形状ひび割れ10m | 2 | 3/3 |

合計8回の点検停止。正解配置から走行後に算出した停止時カメラ前方距離は2.8336～2.8507m（目標2.8m）、停止確認から再開まで3.000～3.057秒でした。画像からの推定距離と正解座標からの距離は区別します。

初期版のCASE_A失敗1回も保存。全非撮影走行では6/7成功、現行モデルでは6/6成功です。認識調整用のcapture走行1回はAIのタスク成功から除外します。根拠は `evaluation_v3/runs.csv` と各 `runs_v3/*/summary.json`、`events.json`、`runtime_model.json`。

## 起動（講座Docker環境）

Windows + Docker Desktop、コンテナ `pai_ros2`、ROS 2 Humble、Ignition Gazebo Fortress 6.16.0、Python 3.10を使用。numpy、cv2、torch、rclpy、cv_bridgeが必要です。学習済みモデルとテクスチャを同梱しているため、モデルの追加ダウンロードは不要です。GUIを使うため、既存のcv2をheadless版に置き換えないでください。

初回は **Windows PowerShell** で、リポジトリ直下からコピーします。既に同名フォルダがある場合は別名で保存してから導入してください。

```powershell
docker start pai_ros2
docker cp .\award_v3 pai_ros2:/home/ubuntu/pai_award_v3
docker exec -u root pai_ros2 chown -R ubuntu:ubuntu /home/ubuntu/pai_award_v3
```

ブラウザの http://localhost:6080/ でUbuntuデスクトップを開き、**Ubuntuターミナル**で実行します。

```bash
cd /home/ubuntu/pai_award_v3
bash scripts/start_all.sh CASE_D
```

`AI wall patrol V3` の画像ウィンドウをクリックし、半角 `s` で開始、`q` で終了。DONEは走行終了です。手動起動ではqでウィンドウを閉じるとGazebo・ブリッジも終了します。録画は別途Snipping Tool等で行います。

自動開始・終了と4条件の連続実行：

```bash
for case_name in CASE_A CASE_B CASE_C CASE_D; do
    bash scripts/start_all.sh "$case_name" auto || break
done
python3 scripts/summarize_runs.py
```

AIタスクの成功は集計結果で確認します。`ROS_DOMAIN_ID=42`、`IGN_PARTITION=pai_award_v3`、画像トピック `/wall/camera` を使用。起動時にワールドのテクスチャ参照を実行環境の絶対パスへ再生成します。`worlds/base.sdf` を直接起動しないでください。

## 認識から動作まで

1. 全クラス共通の青い枠を抽出し、内側の四角形を射影変換します。
2. 枠を除く領域をグレースケール化・96×96画素化・平均輝度補正して3層CNNでnormal / stain / crackを分類。最大出力0.85未満はuncertainです。
3. 枠内の既知高さ1.4m、画像上の高さ、カメラ視野角とyaw=0.45radから前方距離を近似します。
4. crack候補の距離中央値が2.8m以下の状態を異なる3フレームで確認し停止指令。停止判断に正解の異常座標やシナリオ配置ファイルを読み込みません。
5. 新しいオドメトリ2回で並進速度0.01m/s未満・旋回速度0.02rad/s未満を確認し点検画像を保存。約3秒待って再出発し、推定異常位置で重複点検を抑制します。
6. 終点x=12.3mでも停止を確認してDONE。開始位置確認・終点・記録位置にはオドメトリを使用します。

近距離のuncertain候補が3フレーム続く場合は `REVIEW_REQUIRED` として停止し、ひび割れ確定には数えません。画像・オドメトリの途絶、接近対象の消失、走行時間超過も停止・異常終了の対象です。これらはシミュレータ上の制御機能です。

## 学習と画像評価

自作の合成画像を学習720枚、validation 180枚、test 180枚に分割し、seedとSHA-256を `data/manifest.json` に保存。画像ハッシュの重複はsplit間でありません。シーン用画像も別seedです。

初期モデルのGazebo試行では描画後の細いひび割れを十分認識できず `near_target_lost` で失敗。撮影した正常・汚れ・ひび割れ各1枚で入力を確認した後、**元の合成学習画像のみ**に縮小・再拡大・ぼけ・ノイズ等を加えて36エポック追加学習しました。Gazebo撮影画像は学習・モデル選択に使用していません。モデル選択にはvalidationの元画像と固定劣化画像を使用。撮影3枚は開発確認であり、独立テストではありません。

| 3分類argmax評価 | 初期モデル | 現行モデル |
| --- | --- | --- |
| 既存test 180枚 | 167/180 (92.8%) | 165/180 (91.7%) |
| 別seedの合成画像600枚・元画像 | 555/600 (92.5%) | 544/600 (90.7%) |
| 同じ600枚に描画劣化を模擬 | 416/600 (69.3%) | 549/600 (91.5%) |

元画像の精度低下と劣化画像への改善の両方を記録しています。600枚の元画像と劣化画像は対応した同じ内容であり、1,200枚の独立画像ではありません。劣化画像で閾値0.85を適用したcrackの採用数は、真のcrack 200枚中40→138枚、非crack 400枚の誤採用は1→1枚でした。3分類正解率・閾値採用・自律走行成功は別の指標です。単一フレームの誤分類は現在もあり、走行成功は全フレームの正分類を意味しません。

詳細は `checks/camera_model/*.json` と `models/camera_model_card.json`。追加学習では720元画像から各エポックに2,160ビューを作りますが、独立した元画像が2,160枚という意味ではありません。

## 点検記録と再現

`runs_v3/<日時>_<コース>_cnn/` に保存します。

| ファイル | 内容 |
| --- | --- |
| inspection_report.html / .json | 点検画像・停止位置・推定異常位置・CNN出力 |
| inspection_01.png / _annotated.png | 停止確認後のカメラ画像と認識表示 |
| inspection_01_crop.png / _input.png | 取得領域と実際の96×96 CNN入力 |
| perception.jsonl | フレームごとの分類・出力・領域・距離 |
| trace.csv / events.json / summary.json | 速度・状態遷移・点検数・終了状態 |
| runtime_model.json | 実行したモデルのSHA-256と設定 |

HTMLは画像を相対参照するためフォルダごと保存します。例：`runs_v3/20261029_115052_132533_CASE_D_cnn/inspection_report.html`。この走行が動画後半のレポート、`20261029_125222_111841_CASE_D_cnn` が動画前半の走行です。

成功条件：DONE、期待箇所の点検漏れ・余分な点検なし。推定異常位置と正解位置との差1m以内で対応付け、正解配置から計算した停止時カメラ前方距離が2.8m±0.6m以内。正解配置は走行後の評価にのみ使用します。距離許容値は本実証用の基準です。

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 tests/offline_scenes.py
python3 scripts/summarize_runs.py
python3 scripts/evaluate_camera_model.py
```

提出用整備時に制御・データ分離・終了処理等の15テスト成功、全1,080画像のハッシュ照合を確認しました。`offline_scenes.py` は幾何投影と簡易移動モデルでありGazebo実走行とは別です。checks内の初期分類や投影結果も現行モデルの実走行結果と区別してください。

学習をやり直す場合は**作業用コピー**で実行します。次のコマンドは重み・ログ・一部評価ファイルを上書きします。記録済みのSHAと変わった場合はGazebo評価も改めて行ってください。

```bash
python3 scripts/train_walls.py
python3 scripts/train_camera_robust.py
python3 scripts/evaluate_camera_model.py
```

追加学習は同梱の `wall_classifier_before_camera_fix.pt` を初期値として使用。環境差で重みファイルのハッシュまで再現することは保証しません。

## 適用範囲と構成

合成コンクリート模様、共通枠、直線経路、固定カメラ幾何を前提とするシミュレーション実証です。実壁面画像での評価、枠なし候補検出、影・目地・未知汚れへの対応は今後の検証対象です。構造物の診断結果として扱いません。

説明資料は `docs/説明資料_PAI最終課題_84140.pdf`。第三者由来のワールドとApache 2.0ライセンスは `THIRD_PARTY_NOTICES.md` と `LICENSES/` を参照してください。
