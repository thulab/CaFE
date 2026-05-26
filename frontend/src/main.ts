import { createApp } from 'vue';
import App from './App.vue';
import './styles.css';
import { configureAuth } from './api/client';
import { bootstrap, getToken, logout } from './stores/auth';

// 让 api/client 通过这两个钩子访问 auth store，避免双向 import 循环。
configureAuth({
  getToken,
  onUnauthorized: () => {
    logout();
    const here = window.location.hash.replace(/^#/, '') || '/';
    // 不要从 /login 自己再跳 /login，否则死循环。
    if (!here.startsWith('/login')) {
      const next = encodeURIComponent('#' + here);
      window.location.hash = `/login?next=${next}`;
    }
  }
});

// 启动期用已有 token 拉一次 /auth/me（若失败会自动 logout），完成后再挂载 App，
// 避免首屏在未知态下错误闪烁登录页。
bootstrap().finally(() => {
  createApp(App).mount('#app');
});
