# checksの位置付け

- camera_model/existing_test_180.json、fresh_native_600.json、fresh_degraded_600.json：初期モデルと現行モデルの画像評価。
- camera_model/gazebo_development_3.json：撮影調整用の3枚による開発確認。独立した未知画像評価ではありません。
- camera_model/rejected_sharpening.json：不採用とした鮮鋭化の実験。公開プログラムには適用していません。
- projected_scene_checks.json、camera_model/projected_scene_checks.json、CASE_*_projected*.png：幾何投影・簡易移動モデルによる確認。Gazebo実走行とは別です。
- classification_test.json、confidence_test.json、development_initial_classification.json等：初期版の合成画像評価・開発記録。現行モデルの成績はcamera_model内を参照してください。

Gazeboの実走行結果は../evaluation_v3/runs.csvと../runs_v3/内にあります。モデルの識別は../models/camera_model_card.jsonと各runtime_model.jsonを使用します。
