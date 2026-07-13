// quiz.js — reusable quiz widget for RL course lessons
(function() {
  document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.quiz-container').forEach(function(quiz) {
      var options = quiz.querySelectorAll('.quiz-option');
      var feedback = quiz.querySelector('.quiz-feedback');
      var answered = false;

      options.forEach(function(opt) {
        opt.addEventListener('click', function() {
          if (answered) return;
          answered = true;
          var isCorrect = opt.dataset.correct === 'true';
          opt.classList.add(isCorrect ? 'correct' : 'wrong');

          if (!isCorrect) {
            options.forEach(function(o) {
              if (o.dataset.correct === 'true') o.classList.add('correct');
            });
          }

          if (feedback) {
            feedback.style.display = 'block';
            feedback.textContent = isCorrect
              ? (opt.dataset.explanation || '正确！')
              : (quiz.querySelector('[data-correct="true"]').dataset.explanation || '再想想。');
          }
        });
      });
    });
  });
})();
