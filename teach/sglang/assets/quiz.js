document.addEventListener("DOMContentLoaded", () => {
  for (const quiz of document.querySelectorAll(".quiz")) {
    const expectedAnswer = quiz.dataset.answer;
    const correctFeedback = quiz.querySelector(".feedback.correct");
    const incorrectFeedback = quiz.querySelector(".feedback.incorrect");

    for (const radio of quiz.querySelectorAll('input[type="radio"]')) {
      radio.addEventListener("change", () => {
        correctFeedback.style.display = "none";
        incorrectFeedback.style.display = "none";
        const feedback = radio.value === expectedAnswer
          ? correctFeedback
          : incorrectFeedback;
        feedback.style.display = "block";
      });
    }
  }
});
