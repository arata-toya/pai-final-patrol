# 第三者由来ファイル

worlds/base.sdfはGazeboのdiff_drive.sdfを改変したV2のシーンを基礎にしています。

- 上流：https://github.com/gazebosim/gz-sim/blob/ign-gazebo6/examples/worlds/diff_drive.sdf
- ライセンス：Apache License 2.0
- ライセンス全文：LICENSES/Gazebo_Apache-2.0.txt
- 変更：カメラ、センサープラグイン、オドメトリ配信、走行コース。build_worlds.pyが旧標識を取り除き、壁面・巡視路・水面と4つの配置を生成します。

合成コンクリート画像はscripts/synthetic.pyで作成し、既存の施設写真や他者のひび割れデータセットを使用していません。CNNはこの合成画像だけで新たに学習し、V1/V2の色標識用の重みは使用していません。

GazeboのPBR albedo_mapの仕様参考：https://gazebosim.org/api/gazebo/6/migrationsdf.html

説明PDFの日本語表示にはNoto Sans CJK由来のサブセットフォントを埋め込んでいます（SIL Open Font License 1.1）。元フォントの通知・ライセンスはLICENSES/Noto_CJK_copyright.txtに収録しています。フォントの単体ファイルは同梱していません。
