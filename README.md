# WeChat-ClawHub
基于微信官方提供的“微信ClawBot”接口完成的聊天助理项目

> 基于微信官方 ClawBot 接口的多功能智能助手中枢——脚本执行、AI 角色对话、智能体接入，一个入口全部搞定。

[English](README_EN.md) | 简体中文

## 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [架构设计](#架构设计)
- [技术栈](#技术栈)
- [目录结构](#目录结构)
- [环境要求](#环境要求)
- [配置](#配置)
- [使用说明](#使用说明)
  - [脚本指令系统](#脚本指令系统)
  - [AI 角色对话](#ai-角色对话)
  - [智能体接入](#智能体接入)
- [开发指南](#开发指南)
- [项目计划](#项目计划)
- [贡献](#贡献)
- [许可证](#许可证)
- [致谢](#致谢)

## 项目简介

**WeChat ClawHub** 是一个运行在微信 ClawBot 之上的多功能智能助手。你在微信聊天框里发的每一条消息，都会被智能路由到合适的处理模块：

- 输入 `/weather 北京`，执行天气查询脚本并返回结果；
- 输入 `/role set 程序员`，AI 立刻切换成程序员角色和你对话；
- 输入 `/agent deploy`，消息被转发给一个真正的智能体（Agent）来执行复杂任务。

底层基于腾讯官方开放的 **iLink Bot API**（`ilinkai.weixin.qq.com`），通过 **微信 ClawBot 插件功能** 接入。与传统的逆向协议或 Hook 方案不同，这是腾讯官方产品，有《微信ClawBot功能使用条款》法律文件背书，合规且不怕封号[reference:8]。

适用场景：

- 个人开发者：把微信变成自己的“开发助手”，随时执行脚本、查信息
- AI 爱好者：在微信里和不同人格的 AI 角色聊天
- 效率工具：通过微信触发智能体完成自动化任务

## 功能特性

### 🖥️ 脚本指令系统
- [x] 指令注册表模式，新增指令零侵入
- [x] 三级匹配（精确 → 模糊 → 兜底帮助菜单）
- [x] 支持 Python / Node.js / Shell 脚本执行
- [ ] 脚本沙箱与权限控制

### 🤖 AI 角色对话
- [x] 多角色人设切换（`/role set <角色名>`）
- [x] 按角色独立维护对话上下文
- [x] 支持 OpenAI / Anthropic / DeepSeek 等多后端
- [ ] AI 记忆系统（长期记忆 + 自动合并）

### 🔗 智能体接入
- [x] 统一 `Agent` 桥接接口
- [x] 兼容 ACP（Agent Client Protocol）智能体
- [ ] OpenClaw 框架 Agent 适配器
- [ ] MCP 工具扩展

## 架构设计

```text
微信客户端
    │
    ▼
┌─────────────────────────────────────────────┐
│              iLink Bot API                   │
│         (ilinkai.weixin.qq.com)              │
└─────────────────────┬───────────────────────┘
                      │ HTTP 长轮询
                      ▼
┌─────────────────────────────────────────────┐
│              消息路由层 (Router)              │
│   ┌─────────┐ ┌─────────┐ ┌─────────────┐  │
│   │ 精确匹配 │ │ 模糊匹配 │ │ 兜底/帮助菜单 │  │
│   └────┬────┘ └────┬────┘ └──────┬──────┘  │
└────────┼───────────┼─────────────┼──────────┘
         │           │             │
         ▼           ▼             ▼
┌──────────┐  ┌──────────┐  ┌──────────────┐
│  脚本模块 │  │ AI 对话  │  │  智能体桥接层 │
│  Script  │  │  Module  │  │    Agent      │
└──────────┘  └──────────┘  └──────┬───────┘
                                    │
                         ┌──────────┼──────────┐
                         ▼          ▼          ▼
                   OpenClaw    ACP Agent   自定义 HTTP
```

**核心设计原则**：三大功能模块共享同一个消息入口和回复出口，通过统一的路由层分发。新增功能模块只需注册路由规则，不修改核心框架。

## 技术栈

- 语言：TypeScript（Node.js 22+）
- 通信：iLink Bot API（HTTP 长轮询）
- LLM 后端：OpenAI / Anthropic / DeepSeek（可扩展）
- 智能体协议：ACP (Agent Client Protocol)
- 配置管理：dotenv + JSON 配置文件
- 测试：Vitest
- 构建：tsup / esbuild

## 目录结构

```text
.
├── src/
│   ├── core/                # 核心框架
│   │   ├── client.ts        # iLink API 封装
│   │   ├── router.ts        # 消息路由与指令分发
│   │   ├── dispatcher.ts    # 调度中心
│   │   └── session.ts       # 会话状态管理
│   ├── modules/
│   │   ├── script/          # 脚本执行模块
│   │   │   ├── registry.ts  # 指令注册表
│   │   │   └── executor.ts  # 脚本执行器
│   │   ├── ai/              # AI 角色对话模块
│   │   │   ├── persona.ts   # 角色管理
│   │   │   ├── memory.ts    # 对话记忆
│   │   │   └── llm.ts       # LLM 接口封装
│   │   └── agent/           # 智能体接入模块
│   │       ├── bridge.ts    # Agent 桥接接口
│   │       └── adapters/    # 各智能体适配器
│   ├── config/              # 配置管理
│   └── index.ts             # 入口
├── scripts/                 # 用户自定义脚本
├── personas/                # 角色配置文件
│   └── default.json         # 默认角色
├── docs/                    # 文档
├── tests/                   # 测试
├── .env.example             # 环境变量示例
├── package.json
└── README.md
```

## 环境要求

- 微信手机客户端 >= 8.0.70（iPhone）/ 最新版（Android）[reference:9]
- 已安装微信 ClawBot 插件
- 一个可用的 LLM API Key（OpenAI / Anthropic / DeepSeek 任选）

## 配置

### 主配置文件

`config.json`（首次运行自动生成，交互式引导）：

| 字段 | 说明 | 默认值 |
|---|---|---|
| `llm.provider` | LLM 提供商 | `deepseek` |
| `llm.model` | 模型名称 | `deepseek-chat` |
| `llm.temperature` | 默认温度 | `0.7` |
| `script.timeout` | 脚本执行超时 | `10000` |
| `script.workDir` | 脚本工作目录 | `./scripts` |
| `agent.enabled` | 是否启用智能体桥接 | `true` |

### 角色配置文件

`personas/<角色名>.json`：

```json
{
  "name": "程序员",
  "systemPrompt": "你是一个资深全栈工程师，擅长用简洁的代码解决问题。回复时直接给代码和关键解释。",
  "temperature": 0.5,
  "model": "deepseek-chat",
  "greeting": "你好，我是你的编程助手，有什么代码问题尽管问。"
}
```

## 使用说明

### 脚本指令系统

#### 内置指令

| 指令 | 说明 |
|---|---|
| `/help` | 显示所有可用指令 |
| `/role list` | 列出所有已注册角色 |
| `/role set <名称>` | 切换当前角色 |
| `/agent list` | 列出已接入的智能体 |
| `/agent use <名称>` | 切换到指定智能体 |

#### 注册自定义脚本指令

在 `scripts/registry.ts` 中添加一行注册代码即可：

```ts
import { register } from './registry';

register({
  cmd: 'weather',
  keywords: ['weather', '天气', '查天气'],
  handler: async (args, ctx) => {
    const city = args[0] || '北京';
    // 调用天气 API ...
    return `📍 ${city} 当前温度 22°C，晴`;
  },
  helpText: '/weather <城市> — 查询天气',
});
```

注册后，用户在微信中发送 `weather 上海` 或 `查天气 上海` 都能触发。

### AI 角色对话

#### 切换角色

```text
/role set 程序员
→ 已切换到角色「程序员」。你好，我是你的编程助手。
```

#### 与角色对话

角色激活后，所有非指令消息都会进入 AI 对话流程：

```text
用户：帮我写一个快速排序
AI（程序员）：```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```
```

#### 创建新角色

在 `personas/` 目录下新建 JSON 文件即可，支持热重载：

```bash
touch personas/心理咨询师.json
```

```json
{
  "name": "心理咨询师",
  "systemPrompt": "你是一位温和、共情的心理咨询师。用非评判的语言倾听，引导对方表达感受。",
  "temperature": 0.8,
  "greeting": "你好，我在这里，你可以随时跟我说说心里话。"
}
```

### 智能体接入

#### 通过 ACP 接入智能体

如果你的智能体已兼容 ACP 协议，可以直接配置启动命令：

```json
{
  "agent": {
    "adapters": [
      {
        "name": "claude-code",
        "type": "acp",
        "command": "npx wechat-acp claude-code"
      },
      {
        "name": "codex",
        "type": "acp",
        "command": "npx wechat-acp codex"
      }
    ]
  }
}
```

#### 实现自定义 Agent

实现 `Agent` 接口即可接入：

```ts
import type { Agent, ChatRequest, ChatResponse } from './modules/agent/bridge';

const myAgent: Agent = {
  async chat(request: ChatRequest): Promise<ChatResponse> {
    const { text, conversationId } = request;
    // 你的智能体逻辑 ...
    return { text: '处理结果' };
  },
};

export default myAgent;
```

`ChatRequest` 包含 `conversationId`（用户标识，可用于维护多轮对话）和 `text`（消息内容）。

## 项目计划

### Phase 1：基础通信层 ✅
- [x] iLink Bot API 封装（5 个核心接口）
- [x] 扫码登录 + 状态轮询
- [x] 长轮询消息接收
- [x] 消息发送（文本 + “正在输入”状态）
- [x] Token 持久化与自动重连

### Phase 2：脚本指令系统 🚧
- [x] 指令注册表
- [x] 三级匹配引擎（精确 → 模糊 → 兜底）
- [x] Python / Node.js / Shell 脚本执行
- [ ] 脚本沙箱与权限控制
- [ ] 定时任务（Cron 表达式）

### Phase 3：AI 角色对话 📋
- [x] 多 LLM 提供商接入（OpenAI / Anthropic / DeepSeek）
- [x] 角色配置管理（JSON 文件 + 热重载）
- [x] 按角色隔离的对话上下文
- [ ] 长期记忆系统
- [ ] 情绪检测与表情回复
- [ ] 语音消息处理（ASR + TTS）

### Phase 4：智能体接入 📋
- [x] Agent 桥接接口定义
- [x] ACP 适配器基础实现
- [ ] OpenClaw Agent 适配器
- [ ] MCP 工具扩展
- [ ] 多智能体并行管理

### Phase 5：管理与扩展 🎯
- [ ] Web 管理面板（角色管理、对话预览、日志查看）
- [ ] 插件系统（运行时加载第三方插件）
- [ ] Docker 部署支持
- [ ] 多账号管理

## 贡献

欢迎任何形式的贡献！

1. Fork 本仓库
2. 创建分支：`git checkout -b feat/your-feature`
3. 提交更改：`git commit -m "feat: add your feature"`
4. 推送分支：`git push origin feat/your-feature`
5. 提交 Pull Request

请确保：

- 通过测试和 lint
- 更新相关文档
- 一个 PR 只做一件事

详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

本项目基于 [MIT](LICENSE) 许可证开源。

## 致谢

- [OpenClaw](https://docs.openclaw.ai) — 开源 AI 智能体框架
- [@tencent-weixin/openclaw-weixin](https://www.npmjs.com/package/@tencent-weixin/openclaw-weixin) — 腾讯官方微信 ClawBot 插件
- [wechat-acp](https://github.com/Supremesir/wechat-acp) — ACP 桥接框架参考
- [PawzoChat](https://github.com/iwyxdxl/PawzoChat) — 人设系统设计参考
