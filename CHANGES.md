# 修改说明

## 2026-01-29 更新

### 1. 搜索结果跳转支持精确定位（长会话也能跳到命中处）
- 后端 FTS 索引新增 `event_id`（稳定事件 ID）与 `line_no`（jsonl 行号）
- 搜索接口返回定位信息，前端点击“跳转到消息”会从命中行附近加载并滚动定位
- 数据库会在启动时检测 schema，不满足则自动重建（以生成 `event_id/line_no`）

### 2. 阅读体验：代码块一键复制 + 展开/折叠全部
- 工具调用/工具结果等代码块右上角新增“复制”按钮（hover 显示）
- 会话元信息区域新增“展开全部/折叠全部”，作用于当前视图内所有折叠块（`<details>`）
- Markdown 渲染新增 DOMPurify sanitize，避免会话内容中的原始 HTML 造成意外渲染

### 3. 交互与导航增强
- URL 持久化当前状态：project/source/session/view/search 参数，支持刷新与分享链接
- 浏览器前进/后退（popstate）尽量还原到 URL 指定的视图/会话/搜索
- 每条消息新增“🔗”按钮，可复制该消息的 permalink（携带 focus=event_id）

### 7. 会话消息 role 过滤
- 会话元信息区域新增 role 复选框过滤（user/assistant/tool/other），只影响会话视图消息列表

### 4. 会话内查找（复用全文搜索）
- `/api/search` 新增 `session_id` 过滤参数
- 前端搜索栏新增“仅当前会话”开关（优先级高于“仅当前项目/类型”）

### 5. 反馈与索引进度（避免阻塞）
- 后端索引改为后台线程执行，新增 `/api/index/status` 提供进度
- 前端遇到 503（首次启动/索引构建中）会显示进度并自动等待重试

### 6. 兼容性
- 后端类型注解兼容 Python 3.9（避免运行时依赖 `typing.NotRequired`/PEP604 union）

## 2026-01-27 更新

### 1. 增加刷新功能
- 在会话元信息区域添加了"🔄 刷新"按钮
- 点击刷新按钮可以重新加载当前会话的内容，无需切换session
- 实现位置：
  - UI: `claude_viewer.html` 第 471-476 行
  - 事件处理: `claude_viewer.html` 第 1103-1111 行

### 2. 优化 Codex 会话标题显示
- 修改了会话列表中 Codex 会话的标题显示逻辑
- 之前：只显示前8个字符（都是"rollout-"，没有区分度）
- 现在：显示完整的时间戳部分（如 "rollout-2026-01-27T06-57-15"）
- 实现位置：`claude_viewer.html` 第 279-295 行

### 3. 更新 Claude Code 标签
- 将项目选择器中的 `[项目]` 标签改为 `[Claude Code]`
- 实现位置：`claude_viewer.html` 第 206-208 行

## 技术细节

### 刷新功能实现
```javascript
// 在 renderSessionMeta() 中添加刷新按钮
<button
  id="refreshSessionBtn"
  class="ml-4 px-3 py-1 text-xs font-medium text-indigo-600 hover:text-indigo-800 hover:bg-indigo-50 rounded-lg transition-colors"
  title="刷新当前会话"
>
  🔄 刷新
</button>

// 事件委托处理点击
sessionMeta.addEventListener("click", async (event) => {
  const refreshBtn = event.target.closest("#refreshSessionBtn");
  if (refreshBtn && sessionMeta.contains(refreshBtn)) {
    if (state.currentSessionId) {
      await loadSession(state.currentSessionId, false);
    }
  }
});
```

### Codex 标题优化
```javascript
// 对于 codex 会话，显示时间戳部分
if (fullSessionId.startsWith("rollout-")) {
  const parts = fullSessionId.split("-");
  if (parts.length >= 5) {
    sessionIdDisplay = escapeHtml(parts.slice(0, 5).join("-"));
  } else {
    sessionIdDisplay = escapeHtml(fullSessionId.slice(0, 30));
  }
} else {
  // claude code 会话：显示前8个字符
  sessionIdDisplay = escapeHtml(fullSessionId.slice(0, 8));
}
```
