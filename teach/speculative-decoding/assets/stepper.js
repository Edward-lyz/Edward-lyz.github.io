/**
 * Step-through animation controller.
 * 
 * Key: mermaid diagrams must be visible when rendered.
 * Strategy: start all steps visible (CSS has them hidden via .ready class),
 * wait for mermaid to render, then hide non-active steps.
 */
(function() {
  function initSteppers() {
    document.querySelectorAll('.stepper').forEach(stepper => {
      // Mark as ready — CSS will now hide non-active steps
      stepper.classList.add('ready');

      const steps = stepper.querySelectorAll('.step-content');
      const prevBtn = stepper.querySelector('[data-action="prev"]');
      const nextBtn = stepper.querySelector('[data-action="next"]');
      const progress = stepper.querySelector('.progress');
      let current = 0;

      function show(idx) {
        steps.forEach((s, i) => {
          if (i === idx) {
            s.style.display = 'block';
            s.style.opacity = '0';
            requestAnimationFrame(() => { s.style.opacity = '1'; });
          } else {
            s.style.display = 'none';
          }
        });
        current = idx;
        prevBtn.disabled = idx === 0;
        nextBtn.disabled = idx === steps.length - 1;
        progress.textContent = (idx + 1) + ' / ' + steps.length;
      }

      prevBtn.addEventListener('click', () => { if (current > 0) show(current - 1); });
      nextBtn.addEventListener('click', () => { if (current < steps.length - 1) show(current + 1); });
      show(0);
    });
  }

  // Wait for mermaid to finish rendering (it replaces <pre class="mermaid"> with SVG)
  // Use a MutationObserver or a simple delay after DOMContentLoaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => setTimeout(initSteppers, 1500));
  } else {
    setTimeout(initSteppers, 1500);
  }
})();
