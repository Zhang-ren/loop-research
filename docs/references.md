# 技术参考和来源

## 架构设计

- 美团 LOOP AutoResearch §3.1–3.4 给出实验闭环、异步状态机、图式并行探索及搜索策略。§4 给出业务知识与渐进调研。§5 给出原始证据、独立复核、经验审批、归因和领域 Harness。逐项对应的原文章节与图链接见[验收清单](acceptance.json)。[美团技术文章](https://tech.meituan.com/2026/10/22/LOOP-AutoResearch.html)
- Karpathy AutoResearch 提供冻结评估、迭代测量及依据测量保留候选的原型。pi-autoresearch 提供既有仓库中的研究说明和测量入口参考。[Karpathy AutoResearch](https://github.com/karpathy/autoresearch) · [pi-autoresearch](https://github.com/davebcn87/pi-autoresearch)
- DVC 的阶段依赖、输出、指标声明和 Snakemake 的隔离工作目录、失败阶段重跑，作为本项目接入协议设计参考。控制器自行执行声明的命令。[DVC 运行阶段](https://doc.dvc.org/user-guide/project-structure/dvcyaml-files) · [Snakemake Shadow rules](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#shadow-rules) · [Snakemake CLI](https://snakemake.readthedocs.io/en/stable/executing/cli.html)
- 前期也对照了开源研发循环、研究代理、代码代理和替代 Harness。本版本的工作角色由 Codex 执行。此组项目为设计对照。[RD-Agent](https://github.com/microsoft/RD-Agent) · [GPT Researcher](https://github.com/assafelovic/gpt-researcher) · [Deep Agents](https://github.com/langchain-ai/deepagents) · [OpenHands](https://github.com/OpenHands/OpenHands) · [AI Scientist v2](https://github.com/SakanaAI/AI-Scientist-v2) · [AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw) · [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) · [Claude Code 程序化入口](https://code.claude.com/docs/en/headless)

## 运行时和随包代码

| 来源 | 在仓库中的用途 |
| --- | --- |
| [OpenAI Codex CLI](https://github.com/openai/codex) · [非交互模式](https://learn.chatgpt.com/docs/non-interactive-mode) · [权限配置](https://learn.chatgpt.com/docs/agent-approvals-security) · [Skills](https://learn.chatgpt.com/docs/build-skills) | 以独立进程运行有界角色。请求与输出按 Schema 验证，保留 JSONL 与用量。 |
| [LangGraph](https://docs.langchain.com/oss/python/langgraph/graph-api) | 表达控制阶段和路由。控制器仍以文件提交持久状态。 |
| [APScheduler 3.x](https://apscheduler.readthedocs.io/en/3.x/userguide.html) · [FileLock](https://py-filelock.readthedocs.io/) · [psutil](https://psutil.readthedocs.io/) | 周期触发，保护本地提交，并识别进程及其启动时间。 |
| [Pydantic](https://docs.pydantic.dev/latest/) · [Typer](https://typer.tiangolo.com/) · [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation) | 校验配置和角色协议，提供 CLI，读取声明和上游元数据。 |
| [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/) · [SQLAlchemy](https://docs.sqlalchemy.org/) · [Alembic](https://alembic.sqlalchemy.org/) · [SQLite FTS5](https://www.sqlite.org/fts5.html) | 记录参数、指标及引用。SQLAlchemy 和 Alembic 支撑本地追踪库，全文索引可由原件重建。 |
| [Git archive](https://git-scm.com/docs/git-archive) · [Git worktree](https://git-scm.com/docs/git-worktree) · [bubblewrap](https://github.com/containers/bubblewrap) | 用 Git 提交和归档固定可复现源码。worktree 是设计对照。Linux 隔离适配使用 bubblewrap 划分训练、预测、打分空间。 |
| [EvoScientist 固定提交](https://github.com/EvoScientist/EvoScientist/tree/04cb45557057b4b5e035426af4b5553296e2f2d2) | 随包复用并窄适配 AutoSkills 的校验和发布，见 [文件来源和摘要](../src/loop_research/_vendor/evo/UPSTREAM.json)。 |
| [EvoSkills 固定提交](https://github.com/EvoScientist/EvoSkills/tree/85ad685d1df5ed1d7fe641564d04bb8a3dec8f28) | 随包复用研究设计、实验、经验指令及 paper-navigator。见 [文件来源](../src/loop_research/_vendor/evoskills/UPSTREAM.json)和[检索文件清单](../src/loop_research/_vendor/evoskills/paper-navigator/UPSTREAM.json)。 |
| [PyAlex](https://github.com/J535D165/pyalex) · [OpenAlex](https://developers.openalex.org/) · [HTTPX](https://www.python-httpx.org/) · [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) | 检索论文元数据、访问和核查外部文档、提取正文。供应商与正文取用方式受网络和开放许可限制。 |
| [Semantic Scholar API](https://www.semanticscholar.org/product/api) · [Jina Reader](https://github.com/jina-ai/reader) · [arXiv API](https://info.arxiv.org/help/api/) | 随包 paper-navigator 的资源与备用路径。该项目的常规检索以 OpenAlex 为主。 |
| [scikit-learn](https://scikit-learn.org/stable/) · [SciPy](https://docs.scipy.org/doc/scipy/) · [NumPy](https://numpy.org/doc/) · [Matplotlib](https://matplotlib.org/stable/) | 数值兼容与内置示例。Matplotlib 由公开论文试点的实验环境提供，用于输出图。 |
| [Python](https://docs.python.org/3.12/) · [uv](https://docs.astral.sh/uv/concepts/projects/sync/) · [Hatchling](https://hatch.pypa.io/latest/) · [pytest](https://docs.pytest.org/) · [Ruff](https://docs.astral.sh/ruff/) | 运行、锁定依赖、构建和开发工具。直接依赖见 [pyproject.toml](../pyproject.toml)，完整环境解析见 [uv.lock](../uv.lock)。 |

随包文件的许可与改动见 [第三方声明](../THIRD_PARTY_NOTICES.md)。

## 公开验收实验的来源

- 内置基线和项目内经验的实测使用 scikit-learn 的公开数据。仓库附带的传感器阈值样例使用独立的合成数据。[Breast Cancer Wisconsin](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) · [手写数字样例](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- PyTorch 词级语言模型提供多文件结构改动及冻结评测案例。固定上游提交与公开语料见链接。[PyTorch examples](https://github.com/pytorch/examples/tree/acc295dc7b90714f1bf47f06004fc19a7fe235c4/word_language_model) · [WikiText](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/)
- MAPIE 用于区间覆盖率和区间宽度的权衡，对照其原生预测区间入口。[MAPIE 固定提交](https://github.com/scikit-learn-contrib/MAPIE/tree/f1cc3d07460b2d7198f557d40d60cd563c51822b)
- Optuna 用于准确率与计算量的原生多目标测试。[Optuna 固定提交](https://github.com/optuna/optuna/tree/40703c51b7b102df65efaf426f12562d3365735f) · [原生多目标教程](https://optuna.readthedocs.io/en/v4.7.0/tutorial/20_recipes/002_multi_objective.html)
- TabZilla 用于从陌生仓库接入公开结构任务，使用 OpenML Sonar 进行多轮候选、独立重放及留出评测。[TabZilla 固定提交](https://github.com/naszilla/tabzilla/tree/dd2f32cee8c404b30f61efa55577572c6680ab99) · [OpenML Sonar](https://www.openml.org/d/40)
