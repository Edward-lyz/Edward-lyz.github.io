/**
 * Simple quiz widget.
 * Usage:
 *   <div class="quiz" data-answer="2">
 *     <h4>问题文本</h4>
 *     <label><input type="radio" name="q1" value="1"> 选项A</label>
 *     <label><input type="radio" name="q1" value="2"> 选项B</label>
 *     <label><input type="radio" name="q1" value="3"> 选项C</label>
 *     <div class="feedback correct">✓ 正确！解释...</div>
 *     <div class="feedback incorrect">✗ 不对。解释...</div>
 *   </div>
 */
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.quiz').forEach(quiz => {
    const answer = quiz.dataset.answer;
    const correctFb = quiz.querySelector('.feedback.correct');
    const incorrectFb = quiz.querySelector('.feedback.incorrect');
    const radios = quiz.querySelectorAll('input[type="radio"]');

    radios.forEach(radio => {
      radio.addEventListener('change', () => {
        correctFb.style.display = 'none';
        incorrectFb.style.display = 'none';
        if (radio.value === answer) {
          correctFb.style.display = 'block';
        } else {
          incorrectFb.style.display = 'block';
        }
      });
    });
  });
});
