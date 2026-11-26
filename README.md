# LOOP Research

使用 Codex CLI 自动研究本地实验仓库。控制器先确认研究目标、数据划分、评分方式和预算，测出基线，再并行探索候选，独立复核结果，最后用冻结测试集验收。覆盖约定的 76 项本地验收。线上与云端相关能力共 8 项在范围外，长时间真实任务在途的无人值守专项暂缓。[验收记录](docs/acceptance.md) · [逐项清单](docs/acceptance.json)

这是基于美团技术团队公开 LOOP AutoResearch 架构的独立实现。实验循环参考 Karpathy AutoResearch。随包使用 EvoScientist 的提案模块与 EvoSkills 的研究方法和检索工具，来源版本、改动和 Apache 2.0 许可见第三方声明。[美团技术文章](https://tech.meituan.com/2026/10/22/LOOP-AutoResearch.html) · [Karpathy AutoResearch](https://github.com/karpathy/autoresearch) · [第三方声明](THIRD_PARTY_NOTICES.md)

[中文 HTML 使用手册](docs/index.html) · [架构实现](docs/architecture.md) · [实验接入协议](.agents/skills/research-loop/references/local-project.md) · [技术参考](docs/references.md) · [迁移与 GitHub 发布](docs/migration.md)

## 五分钟验证安装

从仓库根目录执行。Linux x86_64、Python 3.12 是本交付实测环境。安装 Git、[uv](https://docs.astral.sh/uv/getting-started/installation/) 和 [Codex CLI](https://learn.chatgpt.com/docs/codex/cli)。使用 Codex 的现有登录，首次运行可执行 `codex login`。Linux 隔离训练还需要主机允许 bubblewrap 使用相应命名空间。

```bash
uv sync --locked --no-editable
codex --version
codex login status
uv run --no-sync loop-research preflight examples/breast-cancer.json
uv run --no-sync loop-research init examples/breast-cancer.json runs/classifier
uv run --no-sync loop-research status runs/classifier
```

`init` 实测基线，不调用规划模型。要启动候选探索，再运行以下命令。训练在本机执行，Codex 角色调用消耗账户额度。

```bash
uv run --no-sync loop-research run runs/classifier
uv run --no-sync loop-research budget runs/classifier
uv run --no-sync loop-research report runs/classifier
```

另有[合成传感器命令适配示例](examples/command-benchmark/RESEARCH-PROJECT.md)，使用 Python 标准库作为训练和评测入口。项目可复用已有训练环境和数据。

## 接入自己的项目

可以从 Codex 使用仓库内的 `$research-loop` 入口 Skill。命令行方式在控制器目录执行，或用本仓库的 `.venv/bin/loop-research` 的绝对路径在任意目录启动。

```bash
uv run --no-sync loop-research prepare-study /path/to/experiment runs/my-study \
  --objective "在既有验证集上提高宏 F1" --non-interactive
```

遇到缺项时查看输出中的 `issues`、`sources` 和 `pending_configuration`。可以重新运行并在终端回答，或用 `--answers /path/to/answers.json` 向同一运行目录补充。检查生成的研究说明、拟合与评测命令、冻结划分、指标方向和阈值、可修改范围与预算。确认后加 `--start` 启动基线及优化。已有完整配置也可用 `loop-research study SPEC RUN`。只测基线时可用 `loop-research init RUN/study-spec.json RUN`。

Linux 原生 Python 项目可用 `loop-research onboard PROJECT RUN --objective "…"` 自动发现并审查入口。加 `--execute --interpreter /abs/path/to/python` 执行基线，加 `--research --rounds 2 --candidates 6` 请求有界优化。需要接续预处理数据时用 `--artifacts` 提供按原仓库相对路径布局的目录。没有经过验证的执行器、独立评测或数据划分时，系统报告具体阻塞项。

## 核心能力

| 功能 | 实现方式 |
| --- | --- |
| 本地预检和补问 | 检查 Git 提交、运行命令、依赖、数据、划分和 GPU 等声明。在启动前记录待补项。 |
| 冻结协议 | 固定代码、种子、指标和划分。先按硬门槛过滤，再按声明的目标优先级排序。 |
| 多轮搜索 | 并行派发有差异的方向，收齐回执后收敛。按适配器支持边界探针、组合重训、源代码探索和条件证伪。 |
| 独立复核 | 按配置、置信度、退化或重要决策触发复核和实际重跑。未解决的分歧进入人工处理。 |
| 调度和恢复 | `tick`、`serve` 或外部事件推进持久状态，租约和训练回执支撑断点恢复。 |
| 项目经验 | 原始证据、自动初筛、独立复核和按哈希批准。只在当前研究检索，支持更新、撤回和回滚。 |
| 证据和复跑 | 保存代码差异、指标、预测、控制与训练回执，验证集用于选择，最终保留集用于验收。 |

Codex 提供独立的角色执行会话，Python 控制器统一管理研究状态、资源和证据。LangGraph 表达阶段路由，文件状态为事实来源。每个研究目录有独立的项目 ID 与检索索引，并行分支有独立会话和工作区。入口 Skill 可以复用，正式研究经验以运行目录为边界。

## 常用命令

- 查看 `status RUN`、`budget RUN`、`directions RUN`、`report RUN`。
- 推进 `tick RUN`、`run RUN`。定时使用 `serve RUN --interval-seconds 60`，事件投递用 `tick RUN --event-id ID`。
- 暂停 `stop RUN`，如需同时取消在途训练加 `--cancel-training`。查询回执用 `reconcile RUN`。
- 核对 `reproduce RUN EVIDENCE_ID`。源码升级后的兼容恢复用 `resume RUN --python /path/to/compatible/python`。
- 查看经验 `memory-list RUN`，复核 `review-memory RUN HASH`，批准 `approve-memory RUN HASH REVIEWER HASH RATIONALE`。
- 测试 `uv run --no-sync pytest -q`。完整的安装、运行与发布流程见 [HTML 手册](docs/index.html)。

可用 `scripts/run_public_paper_study.py --help` 查看公开选题、论文来源、上游复跑、候选比较和论文草稿的额外试点入口。日常运行 V3 研究工作流无需执行该脚本。该试点使用公开数据和外部学术来源。[查看试点用法](docs/index.html#operate)

## 源码与发布

`src/loop_research/` 是控制器，`tests/` 是回归测试，`examples/` 是可分享示例，`.agents/skills/` 是 Codex 入口。`uv.lock` 固定控制器依赖。实验环境与运行回执分别管理。

第三方文件的许可、固定提交和改动见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本项目自有部分的开源许可由维护者选择。
