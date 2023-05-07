## Ham Radio license exam quiz generator

I'm styding for my General class exam and made this to help.

The questions are written in text files in a human-readable form almost identical to the FCC format.

The `test.py` script parses these files an presents you with quizzes.
You can choose to be quizzed on a given "subelement", as they call them,
or on the entire question pool.

Questions are chosen randomly, but the likelihood that a question will be chosen increases if you get it wrong.
As you answer the same question correctly, the likelihood that it will be chosen returns to normal.

![Screenshot](ham-tests.png)

Still a work in progress: I have the rest of the General question set to add.

