# 超景深显微镜共享预约网页部署说明

## 文件位置

网页位于 `docs/`：

- `index.html`：页面结构
- `styles.css`：页面样式
- `app.js`：预约、本地统计、导出逻辑

## GitHub Pages 部署

本仓库已包含 `.github/workflows/pages.yml`。推送到 GitHub 后：

1. 打开仓库 `Settings`。
2. 进入 `Pages`。
3. 将 `Build and deployment` 的 `Source` 设为 `GitHub Actions`。
4. 推送到 `main` 或 `master` 后会自动发布 `docs/` 目录。

也可以不使用 workflow，直接在 `Pages` 中选择 `Deploy from a branch`，分支选择 `main` 或 `master`，目录选择 `/docs`。

## 需要替换的信息

在 `docs/app.js` 中修改：

```js
const ADMIN_PASSCODE = "micro2026";
const ADMIN_EMAIL = "microscope@example.com";
const REMOTE_EVENT_ENDPOINT = "";
```

- `ADMIN_PASSCODE`：统计面板口令，仅用于前端本地查看，不是安全认证。
- `ADMIN_EMAIL`：预约提交后打开邮件草稿的收件邮箱。
- `REMOTE_EVENT_ENDPOINT`：可选远程统计接口。留空时使用浏览器本地存储。

在 `docs/index.html` 中同步替换页面展示的邮箱和电话。

## 统计数据说明

当前版本适配 GitHub Pages 纯静态托管，浏览量、点击量和预约记录默认保存到访问者当前浏览器的 `localStorage`。管理员可在页面统计区域输入口令后查看并导出 CSV。

如果需要全院统一汇总数据，可将 `REMOTE_EVENT_ENDPOINT` 指向 Supabase Edge Function、Cloudflare Worker、Firebase Function 或自有后端接口，在接口中写入数据库。
