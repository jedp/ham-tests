## Ham Radio license exam quiz generator

I'm styding for my General class exam and made this to help.

The questions are taken verbatim from http://www.arrl.org/question-pools.

The `test.py` script parses these files an presents you with quizzes. You can
choose to be quizzed on a given subelement or on the entire question pool.

This is a curses-based, command-line tool. It *should* work on any normal shell.

Questions are chosen at random, but the likelihood that a question will be
selected increases if you answer incorrectly. As you answer the same question
correctly, the likelihood that it will be chosen returns to normal.

### Menu screen

Scroll through the various quizzes you can take:

![menu screen](menu-screen.png)

### Quiz screen

Showing an incorrect answer.

When you choose an answer, it turns white. When you commit to your choice
with the enter key, correct answers turn green; incorrect answers turn red
with the correct answer in green, as shown here:

![quiz screen](quiz-screen-incorrect.png)

