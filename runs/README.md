# Run artifacts

仓库只跟踪可复用的运行模板和简短摘要，不跟踪每次运行生成的完整 `code.py`、临时日志、缓存或大图。

- `templates/`：复制后填写的运行记录模板；
- `summaries/`：已完成实验的短摘要；
- 具体运行目录：默认保持本地，不进入 Git。

成功完成场景选择或论文图生成后，程序会在结果目录自动写入 `run-select-sites.json` 或 `run-generate-figures.json`。该 JSON 是逐次运行的机器可读记录；需要公开某次实验时，把关键字段整理到 `summaries/`，不要提交大型结果和完整缓存。
