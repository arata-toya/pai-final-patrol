# 第三者由来ファイルと変更点

`worlds/patrol.sdf`はGazebo公式の`examples/worlds/diff_drive.sdf`（ign-gazebo6）を改変したものです。

- 上流：https://github.com/gazebosim/gz-sim/blob/ign-gazebo6/examples/worlds/diff_drive.sdf
- 上流リポジトリのライセンス：Apache License 2.0
- ライセンス文：`LICENSES/Gazebo_Apache-2.0.txt`
- 変更：vehicle_blueへのカメラ追加、Sensorsプラグイン追加、緑・赤の点検標識追加、vehicle_green削除。

Qiita記事はROS 2とGazeboの連携方法を参考にしました。記事内のCADファイルやC++のteleopコードは本パッケージに含めていません。
