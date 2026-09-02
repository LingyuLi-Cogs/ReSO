# HarmBench 本地数据

本目录包含 OpenRT 文本攻击评测所需的预下载数据：

- 文件：`harmbench_behaviors_text_test.csv`
- 来源 commit：`centerforaisafety/HarmBench@8e1604d1171fe8a48d8febecd22f600e462bdcdd`
- 路径：`data/behavior_datasets/harmbench_behaviors_text_test.csv`
- SHA256：`75d257b3e7428c52eb7b0154318f455af3e01b09a3794b5e2f3d36054f3c0e29`
- 总行数：320
- 分布：159 standard、81 contextual、80 copyright
- 许可：MIT，见 `LICENSE.HarmBench`

评测器只读取本目录，不会联网下载。若需要重新准备完全相同的文件，可在有网络的
准备机上执行：

```bash
curl -L --fail \
  https://raw.githubusercontent.com/centerforaisafety/HarmBench/8e1604d1171fe8a48d8febecd22f600e462bdcdd/data/behavior_datasets/harmbench_behaviors_text_test.csv \
  -o openrt_text/data/harmbench_behaviors_text_test.csv
shasum -a 256 openrt_text/data/harmbench_behaviors_text_test.csv
```

将整个 `openrt_text/` 与本地 checkpoint 一起复制到离线机器即可运行。
