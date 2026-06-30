import DefaultTheme from 'vitepress/theme'
import { h, nextTick, watch } from 'vue'
import { useRoute } from 'vitepress'
import './custom.css'

export default {
  extends: DefaultTheme,
  Layout() {
    if (typeof window !== 'undefined') {
      const route = useRoute()

      watch(
        () => route.path,
        () => {
          nextTick(() => {
            const renderSiteMath = window.renderSiteMath
            if (typeof renderSiteMath === 'function') {
              renderSiteMath()
            }
          })
        },
        { immediate: true }
      )
    }

    return h(DefaultTheme.Layout)
  }
}
