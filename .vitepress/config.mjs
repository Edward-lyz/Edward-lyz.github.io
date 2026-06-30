import { defineConfig } from 'vitepress'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { existsSync, readdirSync, readFileSync } from 'node:fs'

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const contentRoot = join(repoRoot, '.github', 'build-content')
const articleRoot = join(contentRoot, '文章')

function articleSidebarItems() {
  if (!existsSync(articleRoot)) {
    throw new Error('Generated article folder not found. Run npm run site:prepare before starting VitePress.')
  }

  return readdirSync(articleRoot, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith('.md') && entry.name !== 'index.md')
    .map((entry) => {
      const title = entry.name.slice(0, -'.md'.length)
      const content = readFileSync(join(articleRoot, entry.name), 'utf8')
      const dateValue = content.match(/^date:\s*([0-9-]+)/m)?.[1] ?? '0000-00-00'
      return { title, dateValue }
    })
    .sort((left, right) => right.dateValue.localeCompare(left.dateValue) || left.title.localeCompare(right.title, 'zh-CN'))
    .map(({ title }) => ({ text: title, link: `/文章/${encodeURI(title)}` }))
}

export default defineConfig({
  lang: 'zh-CN',
  title: 'lyz的博客',
  description: 'AI Infra、推理引擎、CUDA 算子与工程思考',
  srcDir: '.github/build-content',
  outDir: 'public',
  cacheDir: '.vitepress/cache',
  base: '/',
  ignoreDeadLinks: [/\.py$/],
  lastUpdated: false,
  sitemap: {
    hostname: 'https://edward-lyz.github.io/'
  },
  markdown: {
    lineNumbers: false,
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    }
  },
  head: [
    ['meta', { name: 'google-site-verification', content: '2dU6Q03zPP1xqF1BcQGKwvDm3P1fH9muifyDNfetdjM' }],
    ['link', { rel: 'preconnect', href: 'https://cdn.jsdelivr.net' }],
    ['link', { rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css' }],
    ['script', { defer: '', src: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js' }],
    ['script', { defer: '', src: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js' }],
    ['script', {}, `
      window.renderSiteMath = function () {
        if (!window.renderMathInElement) return;
        window.renderMathInElement(document.body, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\\\(', right: '\\\\)', display: false },
            { left: '\\\\[', right: '\\\\]', display: true }
          ],
          throwOnError: false
        });
      };
      window.addEventListener('load', window.renderSiteMath);
    `]
  ],
  themeConfig: {
    nav: [
      { text: '文章', link: '/文章/' },
      { text: 'GitHub', link: 'https://github.com/Edward-lyz' }
    ],
    sidebar: {
      '/文章/': [
        {
          text: '文章',
          collapsed: false,
          items: articleSidebarItems()
        }
      ]
    },
    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            displayDetails: '显示详情',
            resetButtonTitle: '清空搜索',
            backButtonTitle: '关闭搜索',
            noResultsText: '没有找到结果',
            footer: {
              selectText: '选择',
              selectKeyAriaLabel: 'enter',
              navigateText: '切换',
              navigateUpKeyAriaLabel: 'up arrow',
              navigateDownKeyAriaLabel: 'down arrow',
              closeText: '关闭',
              closeKeyAriaLabel: 'escape'
            }
          }
        }
      }
    },
    outline: {
      level: [2, 3],
      label: '目录'
    },
    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },
    darkModeSwitchLabel: '主题',
    sidebarMenuLabel: '菜单',
    returnToTopLabel: '回到顶部',
    langMenuLabel: '语言',
    externalLinkIcon: true,
    footer: {
      message: '无限进步',
      copyright: 'Copyright © 2026 lyz'
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/Edward-lyz' }
    ]
  }
})
