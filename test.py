#!/usr/bin/env python3
"""
Ham radio test.

The data files in the pools/ directory are taken from http://www.arrl.org/question-pools.

This program parses them and offers to quiz you on specific subelements,
or all the questions together.

The quiz generator keeps track of errors and increases the likelihood it will select
questions you are wobbly on. As you answer questions correctly with regularity,
the likelihood of being asked those questions returns to normal.
"""

import curses
import os
import random
import re
import sys

from dataclasses import dataclass
from enum import Enum, auto
from textwrap import fill, wrap

class TokenType(Enum):
    """
    Token types for exam text.
    """
    TEXT = 1
    NUMBER = auto()
    PERIOD = auto()
    SPACE = auto()
    HYPHEN = auto()
    EOL = auto()
    OPEN_PAREN = auto()
    CLOSE_PAREN = auto()
    OPEN_BRACKET = auto()
    CLOSE_BRACKET = auto()
    INTERSTICE = auto()
    EOF = auto()


@dataclass
class Token:
    """
    Container for a token and its text content.
    """

    type: TokenType
    text: str

    def __str__(self) -> str:
        return f"<{self.type}: [{self.text}]>"

class Tokenizer:
    """
    Reads a test text file and tokenizes it.

    Be aware that the files have some microsoft-wordy characters in them.
    They may need some massaging.

    Methods to get, peek, and unget.
    """

    def __init__(self, filename: str) -> None:
        self.tokens: list[Token] = []
        self.index = 0
        self.last = 0
        # This looks pretty inefficient.
        tok_p = re.compile(r"(-{2,}|~{2,}|\W)")
        num_p = re.compile(r"^\d+$")
        ws_p = re.compile(r"(\s+)")
        with open(filename, "r") as f:
            for line in f.readlines():
                for word in re.split(tok_p, line):
                    if word in ['\n', '\r']:
                        self.tokens.append(Token(TokenType.EOL, "\n"))
                    elif word == ' ':
                        self.tokens.append(Token(TokenType.SPACE, word))
                    elif word == '.':
                        self.tokens.append(Token(TokenType.PERIOD, word))
                    elif word == '-' or word == '–':
                        self.tokens.append(Token(TokenType.HYPHEN, word))
                    elif word == '(':
                        self.tokens.append(Token(TokenType.OPEN_PAREN, word))
                    elif word == ')':
                        self.tokens.append(Token(TokenType.CLOSE_PAREN, word))
                    elif word == '[':
                        self.tokens.append(Token(TokenType.OPEN_BRACKET, word))
                    elif word == ']':
                        self.tokens.append(Token(TokenType.CLOSE_BRACKET, word))
                    elif word == '~~':
                        self.tokens.append(Token(TokenType.INTERSTICE, word))
                    elif word == '~~~':
                        self.tokens.append(Token(TokenType.EOF, word))
                    elif re.match(num_p, word):
                        self.tokens.append(Token(TokenType.NUMBER, word))
                    elif re.match(ws_p, word):
                        self.last -= 1
                    elif word:
                        self.tokens.append(Token(TokenType.TEXT, word))
                    else:
                        self.last -= 1

                    self.last += 1

    def peek(self) -> Token:
        if self.index >= self.last:
            return Token(TokenType.EOF, "")
        return self.tokens[self.index]

    def get(self) -> Token:
        if self.index >= self.last:
            return Token(TokenType.EOF, "")
        token = self.tokens[self.index]
        self.index += 1
        return token

    def next(self) -> Token:
        if self.index >= self.last:
            return Token(TokenType.EOF, "")
        return self.tokens[self.index + 1]

    def unget(self, positions = 1) -> None:
        self.index -= positions
        assert(self.index >= 0)

    def eat(self, token_type: TokenType) -> Token:
        tok = self.get()
        assert(tok.type is token_type)
        return tok

    def eat_through(self, token_type: TokenType) -> Token:
        self.get_tokens_until(token_type)
        return self.eat(token_type)

    def get_tokens_until(self, token_type: TokenType) -> list[Token]:
        if self.peek().type == TokenType.EOF:
            return []
        tokens = []
        while self.peek().type != token_type:
            tokens.append(self.get())
        return tokens

    def get_text_until(self, token_type: TokenType) -> str:
        return "".join([token.text for token in self.get_tokens_until(token_type)])

    def consume_ws(self) -> int:
        consumed = 0
        while self.index != self.last and self.tokens[self.index].type in [TokenType.SPACE, TokenType.EOL]:
            consumed += 1
            self.index += 1
        return consumed

    def consume_spaces(self) -> int:
        consumed = 0
        while self.index != self.last and self.tokens[self.index].type is TokenType.SPACE:
            consumed += 1
            self.index += 1
        return consumed


class Option(Enum):
    """Multiple choice selection options"""
    A = "A"
    B = "B"
    C = "C"
    D = "D"


@dataclass
class FormattedQuestion:

    number: str
    question: str
    correct: str
    a: str
    b: str
    c: str
    d: str


class Question:
    """
    Contents of a ham radio exam question:
    - Number (e.g., G1A08)
    - Chapter and verse (e.g., 97.303(h))
    - Question text
    - Group of four multiple choice questions (options)
    - The letter of the correct choice
    """

    def __init__(self) -> None:
        self.number = None
        self.question = None
        self.chapter_and_verse = ""
        self.options: list[tuple[Option, str]] = []
        self.correct: Option = Option.A

    def get_subelement_name(self) -> str:
        return self.number[:2]

    def set_number(self, text: str) -> None:
        self.number = text.strip()

    def set_question(self, text: str) -> None:
        self.question = text.strip()

    def set_chapter_and_verse(self, text: str) -> None:
        self.chapter_and_verse = text.strip()

    def add_option(self, label: Option, text: str) -> None:
        assert(len(self.options) <= 4)
        for existing, _ in self.options:
            assert(label is not existing)
        self.options.append((label, text))

    def set_correct(self, label: Option) -> None:
        self.correct = label

    def is_valid(self) -> bool:
        if not self.number:
            return False
        if not self.question:
            return False
        if len(self.options) != 4:
            return False
        if not self.correct:
            return False
        return True

    def formatted(self) -> str:
        "Printable formatted text representation. Like in the book."
        out = f"[{self.number}] {self.chapter_and_verse}\n"
        out += fill(self.question) + "\n\n"
        for label, text in self.options:
            out += "\n".join(wrap(f"  {label.name}. {text}", subsequent_indent='      ')) + "\n"
        return out

    def elements_formatted(self) -> FormattedQuestion:
        return FormattedQuestion(
            number = self.number,
            question = fill(self.question) + "\n\n",
            correct = self.correct.value,
            a = "\n".join(wrap(f"   {self.options[0][0].name}. {self.options[0][1]}", subsequent_indent='       ')) + "\n",
            b = "\n".join(wrap(f"   {self.options[1][0].name}. {self.options[1][1]}", subsequent_indent='       ')) + "\n",
            c = "\n".join(wrap(f"   {self.options[2][0].name}. {self.options[2][1]}", subsequent_indent='       ')) + "\n",
            d = "\n".join(wrap(f"   {self.options[3][0].name}. {self.options[3][1]}", subsequent_indent='       ')) + "\n"
        )


@dataclass
class Subelement:

    name: str
    title: str
    questions: int
    groups: int

    def contains(self, question: Question) -> bool:
        return question.number.name.startswith(self.name)

class QuestionPool:
    """
    A collection of [Question]s.
    """

    def __init__(self) -> None:
        self.questions: list[Question] = list()
        self.subelements: list[Subelement] = list()

    def __len__(self) -> int:
        return len(self.questions)

    def add_subelement(self, subelement: Subelement) -> None:
        self.subelements.append(subelement)

    def add_question(self, question: Question) -> None:
        self.questions.append(question)

    def get_elements(self) -> list[Subelement]:
        return self.subelements

    def get_question_at(self, index: int) -> Question:
        return self.questions[index]

    def get_questions_for_element(self, element: Subelement) -> list[Question]:
        return [q for q in self.questions if q.number.startswith(element.name)]



class State(Enum):
    """Question Parser state machine states"""
    ERROR = -1
    START = 0
    PREAMBLE = 1
    GROUP_OR_QUESTION = 2
    GROUP = 3
    QUESTION = 4
    OPTION = 5
    DONE = 6



class QuestionParser:

    def __init__(self, question_pool: QuestionPool, file_path: str) -> None:
        self.file_path = file_path
        self.tokens = Tokenizer(file_path)
        self.subelement = None
        self.current_question = None
        self.current_group = ""
        self.error = ""
        self.state = State.START
        self.questions = question_pool

    def fail(self, msg: str):
        raise Exception(msg)

    def expect(self, condition, msg):
        if not condition:
            self.fail(msg)

    def parse(self) -> None:
        while True:
            match self.state:
                case State.ERROR:
                    raise Exception(self.error)

                case State.START:
                    "Start parsing from the beginning of a file"
                    self.state = State.PREAMBLE

                case State.PREAMBLE:
                    """
                    The first line of the file is like:
                    SUBELEMENT G1 – COMMISSION’S RULES [5 Exam Questions – 5 Groups]
                    """
                    self.expect(self.tokens.get_text_until(TokenType.SPACE) == "SUBELEMENT",
                                "Expected SUBELEMENT in preamble")
                    self.tokens.consume_ws()
                    name = self.tokens.get_text_until(TokenType.SPACE).strip()

                    self.tokens.consume_ws()
                    self.tokens.eat(TokenType.HYPHEN)
                    self.tokens.consume_ws()

                    title = self.tokens.get_text_until(TokenType.OPEN_BRACKET).strip()
                    self.tokens.eat(TokenType.OPEN_BRACKET)
                    question_count = int(self.tokens.get().text)
                    self.expect(0 < question_count, f"Invalid question count: {question_count}")
                    self.tokens.get_tokens_until(TokenType.NUMBER)
                    group_count = int(self.tokens.get().text)
                    self.expect(0 < group_count, f"Invalid group count: {group_count}")

                    self.tokens.eat_through(TokenType.EOL)

                    self.subelement = Subelement(
                        name = name,
                        title = title,
                        questions = question_count,
                        groups = group_count
                    )
                    self.questions.add_subelement(self.subelement)
                    self.state = State.GROUP

                case State.GROUP_OR_QUESTION:
                    """
                    Determine whether the next object to parse is a Group name or a Question
                    or whether we have reached the end of the file.
                    """
                    next_tok = self.tokens.peek()
                    if next_tok.type is TokenType.EOF:
                        self.state = State.DONE
                    elif next_tok.text == "SUBELEMENT":\
                        self.state = State.PREAMBLE
                    elif len(next_tok.text) == 5 and next_tok.text.startswith(self.current_group):
                        self.state = State.QUESTION
                    else:
                        self.state = State.GROUP

                case State.GROUP:
                    "Update the current group name"
                    name = self.tokens.get_text_until(TokenType.SPACE).strip()
                    self.expect(name.startswith(self.subelement.name), f"Invalid group name: {name}")
                    self.current_group = name
                    self.tokens.eat(TokenType.SPACE)
                    self.tokens.eat(TokenType.HYPHEN)
                    self.tokens.eat(TokenType.SPACE)
                    title = self.tokens.get_text_until(TokenType.EOL).strip()
                    self.tokens.eat(TokenType.EOL)
                    # Maybe we will want this in the future.
                    # Add a title to it or something.
                    self.state = State.QUESTION

                case State.QUESTION:
                    """
                    Parse a question, which is like:

                    G1A02 (B) [97.305]
                    On which of the following bands is phone operation prohibited?
                    A. 160 meters
                    B. 30 meters
                    C. 17 meters
                    D. 12 meters
                    """

                    # The question key, a sequence like G1A02
                    self.current_question = Question()
                    key = self.tokens.get_text_until(TokenType.SPACE).strip()
                    self.expect(key.startswith(self.current_group), f"Out-of-order question key: {key}")
                    self.current_question.set_number(key)

                    # The correct answer
                    self.tokens.eat_through(TokenType.OPEN_PAREN)
                    answer = self.tokens.get().text
                    self.expect(answer in ['A', 'B', 'C', 'D'], f"Weird correct answer: {answer}")
                    self.current_question.set_correct(Option(answer))
                    self.tokens.eat(TokenType.CLOSE_PAREN)
                    self.tokens.consume_spaces()

                    # Maybe chapter and verse
                    chapter_and_verse = self.tokens.get_text_until(TokenType.EOL)
                    self.tokens.eat(TokenType.EOL)
                    self.current_question.set_chapter_and_verse(chapter_and_verse)

                    # Read the question
                    question = self.tokens.get_text_until(TokenType.EOL).strip()
                    self.expect(len(question) > 0, "Empty question text")
                    self.tokens.eat(TokenType.EOL)
                    self.current_question.set_question(question)
                    self.questions.add_question(self.current_question)

                    self.state = State.OPTION

                case State.OPTION:
                    "Parse one of the multiple choice options"
                    letter = self.tokens.get().text
                    self.expect(letter in ['A', 'B', 'C', 'D'], f"Bad option letter: {letter}")
                    self.tokens.eat(TokenType.PERIOD)
                    text = self.tokens.get_text_until(TokenType.EOL)
                    self.tokens.eat(TokenType.EOL)

                    self.current_question.add_option(Option(letter), text)
                    # Interstice follows last letter
                    if letter == 'D':
                        self.expect(self.current_question.is_valid(), "Question not converted successfully")
                        self.tokens.eat(TokenType.INTERSTICE)
                        self.tokens.eat_through(TokenType.EOL)
                        self.state = State.GROUP_OR_QUESTION

                case State.DONE:
                    "Done reading this file"
                    print(f"Read {len(self.questions)} questions from {self.file_path}.")
                    return

                case _:
                    raise Exception(f"Unhandled state: {self.state}")


class ScoreKeeper:
    """
    Persist error count for question weighting.

    A higher score means more errors. Minimum score is 0.

    Each wrong answer adds an error to the count. Each right answer subtracts one.
    """

    def __init__(self) -> None:
        self.datafile = os.path.join(os.getcwd(), ".ham-test-weights.dat")
        self.errors = {}

    def get_scores(self, questions: list[Question]) -> dict:
        """
        Return a dictionary of errors keyed by question number.

        If the data file exists, load existing scores from it, adding a 0 for
        any new values.

        If the data file doesn't exist, don't write anything. We can do that
        when we save at the end.
        """
        if len(self.errors) == 0:
            if os.path.exists(self.datafile):
                with open(self.datafile, 'r') as f:
                    self.errors = eval(f.read())
        # Supplement any missing values
        for q in questions:
            if q.number not in self.errors:
                self.errors[q.number] = 0
        return self.errors

    def store_scores(self) -> None:
        """
        Write the current dictionary of errors to disk.
        """
        with open(self.datafile, "w") as f:
            f.write(str(self.errors))

    def right_answer(self, question: Question) -> int:
        if question.number not in self.errors:
            self.errors[question.number] = 0
        else:
            self.errors[question.number] = max(self.errors[question.number] - 1, 0)
        return self.errors[question.number]

    def wrong_answer(self, question: Question) -> int:
        if question.number not in self.errors:
            self.errors[question.number] = 1
        else:
            self.errors[question.number] = self.errors[question.number] + 1
        return self.errors[question.number]


class ChoiceColor(Enum):
    NO = 1
    YES = 2

class RecentQuestions(list):

    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.capacity = capacity

    def include(self, item: any) -> None:
        if item not in self:
            self.insert(0, item)
        if len(self) > self.capacity:
            self.pop()

class CursesApp:

    def __init__(self, exam_level: str) -> None:
        self.stdscr = None
        self.question_pool: QuestionPool = QuestionPool()
        self.scorekeeper = ScoreKeeper()
        self.recent_questions = RecentQuestions(10)
        self.subelement_question_map = {}

        # Load questions
        cwd = os.getcwd()
        filepath = os.path.join(cwd, 'pools', exam_level+'.txt')
        question_parser = QuestionParser(self.question_pool, filepath)
        question_parser.parse()

        for subelement in self.question_pool.subelements:
            self.subelement_question_map[subelement.name] = []

        for question in self.question_pool.questions:
            self.subelement_question_map[question.get_subelement_name()].append(question)

    def _load_questions(self, exam_level: str) -> None:
        cwd = os.getcwd()
        filepath = os.path.join(cwd, 'pools', exam_level+'.txt')
        question_parser = QuestionParser(self.question_pool, filepath)
        question_parser.parse()

        for subelement in self.question_pool.subelements:
            self.subelement_question_map[subelement.name] = []

        for question in self.question_pool.questions:
            self.subelement_question_map[question.get_subelement_name()].append(question)

    def _select_question(self, question_set: list[Question]) -> Question:
        """
        Weighted random choice: Questions that have been answered wrong are
        more likely to be chosen than other questions.
        """
        error_counts = self.scorekeeper.get_scores(question_set)
        assert(len(question_set) > 10)
        while True:
            q = random.choices(
                population = question_set,
                weights = [error_counts[q.number] * 3 + 1 for q in question_set],
                k = 1
            )[0]
            if q.number not in self.recent_questions:
                self.recent_questions.include(q.number)
                return q

    def _guess(self, question: Question, letter: Option) -> Option:
        if question.correct == letter:
            self.scorekeeper.right_answer(question)
        else:
            self.scorekeeper.wrong_answer(question)
        return question.correct

    def _quiz(self, question_set: list[Question]) -> None:
        self.stdscr.clear()
        self.stdscr.refresh()
        choices = '[a, b, c, d, q]'
        # Keep track of colors for the four options.
        # They'll be highlighted when selected,
        # And red or green for wrong/right guesses.
        colors = {
            'A': curses.A_NORMAL,
            'B': curses.A_NORMAL,
            'C': curses.A_NORMAL,
            'D': curses.A_NORMAL
        }

        action = -1   # getch()
        selected = "" # Selected option

        num_questions = len(question_set)
        question = self._select_question(question_set)
        formatted = question.elements_formatted()

        while action != ord('q'):
            question_number = question_set.index(question) + 1
            self.stdscr.clear()
            # Print the question
            row = 3
            self.stdscr.addstr(row, 0, formatted.number)
            self.stdscr.addstr(row, 58, f"{question_number} / {num_questions}")
            row += 1
            self.stdscr.addstr(row, 0, formatted.question)
            row += formatted.question.count("\n")
            # Print the four options
            self.stdscr.addstr(row, 0, formatted.a, colors['A'])
            row += formatted.a.count("\n")
            self.stdscr.addstr(row, 0, formatted.b, colors['B'])
            row += formatted.b.count("\n")
            self.stdscr.addstr(row, 0, formatted.c, colors['C'])
            row += formatted.c.count("\n")
            self.stdscr.addstr(row, 0, formatted.d, colors['D'])
            row += formatted.d.count("\n")
            row +=2
            # Print a menu of choices
            self.stdscr.addstr(row, 0, choices)

            self.stdscr.refresh()
            action = self.stdscr.getch()

            if ord('a') <= action <= ord('d'):
                # Selected a, b, c, or d
                selected = chr(action).upper()
                choices = '[a, b, c, d, q; Enter to select]'
                colors = dict.fromkeys(colors, curses.A_NORMAL)
                colors[selected] = curses.A_STANDOUT

            elif selected and (action == curses.KEY_ENTER or action == 10 or action == 13):
                # Hit enter to confirm selection
                if question.correct.value == selected:
                    self.scorekeeper.right_answer(question)
                    colors[selected] = curses.color_pair(ChoiceColor.YES.value)
                else:
                    self.scorekeeper.wrong_answer(question)
                    colors[selected] = curses.color_pair(ChoiceColor.NO.value)
                    colors[question.correct.value] = curses.color_pair(ChoiceColor.YES.value)
                choices = '[q; n for Next]'

            elif action == ord('n'):
                # Next question
                question = self._select_question(question_set)
                formatted = question.elements_formatted()
                # Reset colors and choices menu
                colors = dict.fromkeys(colors, curses.A_NORMAL)
                choices = '[a, b, c, d, q]'

    def main(self, stdscr) -> None:
        """
        Main entry point.

        When invoking, wrap with curses wrapper.
        """
        self.stdscr = stdscr

        curses.cbreak()
        curses.noecho()
        curses.curs_set(False)
        curses.start_color()
        curses.init_pair(ChoiceColor.NO.value, curses.COLOR_BLACK, curses.COLOR_RED)
        curses.init_pair(ChoiceColor.YES.value, curses.COLOR_BLACK, curses.COLOR_GREEN)
        stdscr.keypad(True)
        stdscr.clear()

        action = -1    # getch()
        menu_rows = [] # This is kind of rude, but it works.
        i = 1
        for subelement in self.question_pool.subelements:
            menu_rows.append(f"[{i}] {subelement.name}: {subelement.title}")
            i += 1
        menu_rows.append("[a] All")
        menu_rows.append("[q] Quit")
        selection = 0

        while action != ord('q'):
            stdscr.clear()

            # keep up/down arrow selection in bounds.
            if selection < 0:
                selection = 0
            if selection > len(menu_rows) - 1:
                selection = len(menu_rows) - 1

            # print all the menu choices
            for i in range(0, len(menu_rows)):
                stdscr.addstr(i + 3, 0, menu_rows[i], (i == selection) and curses.A_STANDOUT or curses.A_NORMAL)

            stdscr.refresh()

            action = stdscr.getch()

            if action == ord('a'):
                selection = len(menu_rows) - 2

            elif ord('1') <= action <= ord('9'):
                # Sadly you can't select '10' this way.
                selection = action - ord('1')

            elif action == curses.KEY_UP:
                # Select previous
                selection -= 1

            elif action == curses.KEY_DOWN:
                # Select next
                selection += 1

            elif action == curses.KEY_ENTER or action == 10 or action == 13:
                if selection == len(menu_rows) - 1:
                    action = ord('q')
                elif selection == len(menu_rows) - 2:
                    # All
                    stdscr.clear()
                    self._quiz(self.question_pool.questions)
                else:
                    # Single subelement
                    stdscr.clear()
                    subelement = self.question_pool.subelements[selection]
                    question_set = [q for q in self.question_pool.questions if q.get_subelement_name() == subelement.name]
                    self._quiz(question_set)

        # Save and quit
        self.scorekeeper.store_scores()


def main() -> int:
    app = CursesApp('general')
    curses.wrapper(app.main)
    return 1


if __name__ == '__main__':
    sys.exit(main())

