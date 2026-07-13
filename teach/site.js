(() => {
  const appearance = localStorage.getItem('vitepress-theme-appearance')
  const followsSystem = appearance !== 'dark' && appearance !== 'light'
  const useDark = appearance === 'dark' || (
    followsSystem && window.matchMedia('(prefers-color-scheme: dark)').matches
  )
  document.documentElement.classList.toggle('dark', useDark)
})()
