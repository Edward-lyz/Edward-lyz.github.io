// 共享测验交互：点击选项即时判分 + 展开解释。
// 所有 lesson 复用，配合 assets/lesson.css 的 .quiz 结构。
function handleQuiz(e, qId) {
  const li = e.target.closest('li');
  if (!li) return;
  const container = document.getElementById(qId);
  const options = container.querySelectorAll('.quiz-options li');
  const explain = container.querySelector('.quiz-explain');
  options.forEach(o => o.classList.remove('correct', 'wrong'));
  if (li.dataset.correct) {
    li.classList.add('correct');
  } else {
    li.classList.add('wrong');
    options.forEach(o => { if (o.dataset.correct) o.classList.add('correct'); });
  }
  explain.classList.add('show');
}
