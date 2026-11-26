# 迁移与 GitHub 发布

## 在新主机安装

1. 安装 Git、uv 和能运行本项目权限档案的 Codex CLI。实测平台是 Linux x86_64、Python 3.12、Codex CLI 0.172.0。Linux 隔离训练另外需要能使用 bubblewrap 的主机环境。macOS 和 Windows 的完整原生训练链未做此次迁移验收。
2. 克隆仓库或解压源码 ZIP。在仓库根目录执行 `uv sync --locked --no-editable`。若缺少 Python 3.12，可以先执行 `uv python install 3.12`。
3. 用 `codex login` 在新环境登录。必要时将 `LOOP_CODEX_BINARY` 设为本机可执行的 Codex 路径。控制器启动有界工作角色，权限控制和工具网络策略由 Codex CLI 执行。角色请求发给配置的模型服务。
4. 复跑 [README 的基线示例](../README.md#五分钟验证安装)。实际实验另装其训练依赖，在接入时声明解释器、数据路径、命令、GPU 和依赖模块。新的研究目录生成独立项目 ID。
5. 独立运行时可用 `uv build` 产出 wheel 和源码包。在目标主机已有锁定环境时，用 `uv sync --locked --no-editable` 获得与仓库相同的依赖集合。`uv tool install /path/to/checkout` 可提供全局入口，其工具环境单独解析依赖。

## 在其他项目使用 Codex 入口

先确保目标项目里运行 `loop-research --help` 可以找到控制器。已有锁定安装时，可将控制器仓库的 `.venv/bin` 加入当前终端的 `PATH`，或使用 CLI 的绝对路径。下面的项目级安装会保留 Skill 和关联协议文档。

```bash
mkdir -p /path/to/experiment/.agents/skills
cp -R .agents/skills/research-loop /path/to/experiment/.agents/skills/
cd /path/to/experiment
codex
```

在 Codex 中用 `$research-loop` 提供目标、允许改动的范围和研究预算。要跨多个项目复用入口，也可安装到 `~/.agents/skills/`。正式实验经验仍从每个研究目录检索。[Codex Skills 的安装与发现](https://learn.chatgpt.com/docs/build-skills)

## 转移已有研究

已归档的研究目录同时包含项目锁定状态、训练与调用账本、预测、代码和审批记录。保存完整目录及其外部依赖，保留原访问权限。训练、评分器、索引和原生适配的某些回执包含绝对路径。恢复既有实验时，应在新机器重建原有路径布局和运行环境，停止旧的控制进程，先核对 `status RUN` 和 `reconcile RUN`，再试 `resume RUN --python /path/to/compatible/python --max-ticks 1`。控制器会校验归档版本和数值依赖。路径发生变化时，可以在新位置重跑基线建立新研究。现有格式未提供任意路径重定位工具，进行中的 PID、租约和作业也无法随文件直接迁移。

发布仓库只保存应用源码和可分发的样例。实验数据、环境目录、账号凭据、完整角色日志、模型权重和已批准的项目经验单独保管。不要将旧研究的经验目录复制进新的研究。

## 首次提交和推送

在 GitHub 创建一个空仓库，再在本项目目录检查提交内容。Git 需已有你的用户名和邮箱。若要以开源项目发布，在首次提交前确定自有源码许可并加入根目录 `LICENSE`。上游文件继续使用它们随附的 Apache 2.0 许可证。[GitHub 仓库许可证说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)

```bash
git status --short
git diff --cached --stat
git add --all
git diff --cached --name-only
git commit -m "Initial release of LOOP Research"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

ZIP 解压后若目录尚未初始化 Git，先执行 `git init -b main`。使用现有 GitHub 仓库时，先在目标仓库的工作副本中检查已有文件，再提交本目录所列内容。[GitHub 导入本地代码说明](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)

浏览器可以直接打开 `docs/index.html`。在 GitHub 可先浏览 README。若希望用网址访问 HTML，可在仓库的 Pages 设置中把 `main` 分支的 `/docs` 作为站点目录，站点正文可离线显示。外部参考和指向仓库根部的链接在本地查看更完整。[GitHub Pages 发布源](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
