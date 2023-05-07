#!/usr/bin/env python3
"""
Tokenizer, parser, and quiz engine for my human-readable and arbitrary
ham radio test question file format. Files look like what you get in
the FCC question format. Example:

```
Subelement: G1
Title: Commission's Rules
Exam Questions: 5
Groups: 5

Group: G1A

G1A01
On which HM/MF bands is a General class license holder granted all amateur
frequency privileges?
--
 A. 60, 20, 17, and 12 meters
 B. 160, 80, 40, and 10 meters
*C. 160, 60, 30, 17, 12, and 10 meters
 D. 160, 30, 17, 15, 12, and 10 meters

G1A02
On which of the following bands is phone operations prohibited?
--
 A. 160 meters
*B. 30 meters
 C. 17 meters
 D. 12 meters
```
etc.

A few features make for nice formatting and much easier automatic parsing.
Two things in particular:

- Questions all end with a "--" on an otherwise empty line.
- Wrapped lines are all indented deeper than the A, B, C, D letters.

"""

import os
import random
import re
import sys

from dataclasses import dataclass
from enum import Enum, auto
from textwrap import fill, wrap

class State(Enum):
    """Question Parser state machine states"""
    ERROR = -1
    START = 0
    SUBELEMENT = 1
    TITLE = 2
    QUESTIONS = 3
    GROUPS = 4
    GROUP_OR_QUESTION = 5
    GROUP = 6
    QUESTION = 7
    OPTION = 8
    DONE = 9


class Option(Enum):
    """Multiple choice selection options"""
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class Question:
    """
    Contents of a ham radio exam question:
    - Number (e.g., G1A08)
    - Question text
    - Group of four multiple choice questions (options)
    - The letter of the correct choice
    """
    
    def __init__(self) -> None:
        self.number = None
        self.question = None
        self.options: list[tuple[Option, str]] = []
        self.correct = None

    def get_subelement_name(self) -> str:
        return self.number[:2]
    
    def set_number(self, text: str) -> None:
        self.number = text.strip()
    
    def set_question(self, text: str) -> None:
        self.question = text.strip()

    def add_option(self, label: Option, text: str) -> None:
        assert(len(self.options) <= 4)
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
        out = f"[{self.number}]\n\n"
        out += fill(self.question) + "\n\n"
        for label, text in self.options:
            out += "\n".join(wrap(f"{label.name}. {text}", subsequent_indent='   ')) + "\n"
        return out

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

class TokenType(Enum):
    """
    Token types for exam text.
    """
    TEXT = 1
    ASTERISK = auto()
    NUMBER = auto()
    PERIOD = auto()
    COLON = auto()
    SPACE = auto()
    EOT = auto()
    EOL = auto()
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
    
    Methods to get, peek, and unget.
    """

    def __init__(self, filename: str) -> None:
        self.tokens: list[Token] = []
        self.index = 0
        self.last = 0
        # This looks pretty inefficient.
        tok_p = re.compile(r"(-{2,}|\W)")
        num_p = re.compile(r"^\d+$")
        ws_p = re.compile(r"(\s+)")
        with open(filename, "r") as f:
            for line in f.readlines():
                for word in re.split(tok_p, line):
                    if word == '\n':
                        self.tokens.append(Token(TokenType.EOL, word))
                    elif word == ' ':
                        self.tokens.append(Token(TokenType.SPACE, word))
                    elif word == '*':
                        self.tokens.append(Token(TokenType.ASTERISK, word))
                    elif word == ':':
                        self.tokens.append(Token(TokenType.COLON, word))
                    elif word == '.':
                        self.tokens.append(Token(TokenType.PERIOD, word))
                    elif word == '--':
                        self.tokens.append(Token(TokenType.EOT, word))
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
        
    def eat(self) -> Token:
        return self.get()
    
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

class QuestionParser:

    def __init__(self, question_pool: QuestionPool, file_path: str) -> None:
        self.file_path = file_path
        self.tokens = Tokenizer(file_path)
        self.subelement = ""
        self.current_option = None
        self.current_question = None
        self.current_subelement_name = ""
        self.current_subelement_title = ""
        self.current_subelement_groups = 0
        self.current_subelement_questions = 0
        self.current_group = ""
        self.error = ""
        self.state = State.START
        self.questions = question_pool
        
    def _read_kv_pair_str(self) -> tuple[str, str]:
        """Convenience to read "Key: Some value" lines"""
        key = self.tokens.get_text_until(TokenType.COLON)
        assert(self.tokens.eat().type == TokenType.COLON)
        self.tokens.consume_ws() 

        value = self.tokens.get_text_until(TokenType.EOL)    
        assert(self.tokens.eat().type == TokenType.EOL)
        self.tokens.consume_ws()
        return key, value
    
    def _read_kv_pair_num(self) -> tuple[str, int]: 
        """Convenience to read "Key: IntVal" lines"""
        key = self.tokens.get_text_until(TokenType.COLON)
        assert(self.tokens.eat().type == TokenType.COLON)
        self.tokens.consume_ws() 

        value_tok = self.tokens.get()
        assert(value_tok.type == TokenType.NUMBER)
        self.tokens.get_tokens_until(TokenType.EOL)
        self.tokens.consume_ws()
        return key, int(value_tok.text)
    
    def fail(self, msg: str):
        raise Exception(msg)

    def parse(self) -> None:
        while True:
            match self.state:
                case State.ERROR:
                    raise Exception(self.error)

                case State.START:
                    "Start parsing from the beginning of a file"
                    self.state = State.SUBELEMENT

                case State.SUBELEMENT:
                    "Read the label of the test subelement (e.g., G1)"
                    label, text = self._read_kv_pair_str()
                
                    if label != "Subelement":
                        self.error = f"Expected Subelement: Got '{label}'"
                        self.state = State.ERROR
                    else:
                        self.current_subelement_name = text
                        self.state = State.TITLE
                        self.tokens.consume_ws()
                    
                case State.TITLE:
                    "Read the title of the test subelement"
                    label, text = self._read_kv_pair_str()

                    if label != "Title":
                        self.error = f"Expected Title: Got '{label}'"
                        self.state = State.ERROR
                    else:
                        self.current_subelement_title = text
                        self.state = State.QUESTIONS
                        self.tokens.consume_ws()

                case State.QUESTIONS:
                    "Read the number of questions to be selected from these groups for the exam"
                    label, count = self._read_kv_pair_num()

                    if label != "Exam Questions":
                        self.error = f"Expected Questions count: Got: {label}: {count}"
                        self.state = State.ERROR
                    else:
                        self.current_subelement_questions = count
                        self.state = State.GROUPS

                case State.GROUPS:
                    "Read the number of groups in this subelement"
                    # Read number of groups
                    label, count = self._read_kv_pair_num()

                    if label != "Groups":
                        self.error = "Expected GROUPS count"
                        self.state = State.ERROR
                    else:
                        self.current_subelement_groups = count
                        self.questions.add_subelement(
                            Subelement(
                                name = self.current_subelement_name,
                                title = self.current_subelement_title,
                                questions = self.current_subelement_questions,
                                groups = self.current_subelement_groups
                            )
                        )
                        self.state = State.GROUP_OR_QUESTION
                        
                case State.GROUP_OR_QUESTION:
                    "Determine whether the next object to parse is a Group name or a Question"
                    question = Question()
                    self.questions.add_question(question)
                    self.current_question = question

                    self.tokens.consume_ws()
                    this_tok = self.tokens.get()
                    next_tok = self.tokens.peek()
                    if next_tok.type == TokenType.EOF:
                        self.state = State.DONE
                    else: 
                        if this_tok.text == "Group" and next_tok.type == TokenType.COLON:
                            self.state = State.GROUP
                        else:
                            self.state = State.QUESTION
                        # Now push back the token we read to figure this out
                        self.tokens.unget()
                
                case State.GROUP:
                    "Update the current group name"
                    label, text = self._read_kv_pair_str()
                    assert(label == "Group")
                    # Maybe we will want this in the future.
                    # Add a title to it or something.
                    self.state = State.QUESTION
                    
                case State.QUESTION:
                    "Parse a question"
                    # Read the ID, which is like G1A01
                    self.tokens.consume_ws()
                    question_id = self.tokens.get().text
                    if not question_id.startswith(self.current_group):
                        self.fail(f"Question ID {question_id} does not start with expected group {self.current_group}")
                    self.tokens.consume_ws()
                    
                    # Read the content of the question up to the '--' (EOT) marker
                    words = []
                    word_tok = self.tokens.get()
                    while word_tok.type != TokenType.EOT:
                        words.append(word_tok.text)
                        word_tok = self.tokens.get()
                        if word_tok.type == TokenType.EOF:
                            self.state = State.DONE
                            break
                    text = "".join(words).strip()

                    self.current_question.set_number(question_id)
                    self.current_question.set_question(text)
                    self.state = State.OPTION
                    
                case State.OPTION:
                    "Parse one of the multiple choice options"
                    # We'll see if this is the correct answer.
                    is_correct = False
                    
                    # Possible leading space.
                    self.tokens.consume_ws()
                    
                    # Maybe this is the right answer.
                    if self.tokens.peek().type is TokenType.ASTERISK:
                        self.tokens.eat()
                        is_correct = True
                        
                    # Read the multiple choice letter.
                    letter = self.tokens.get().text
                    if letter not in ['A', 'B', 'C', 'D']:
                        self.fail(f"Expected multiple choice letter; got: {letter}")
                    assert(self.tokens.eat().type is TokenType.PERIOD)
                    self.tokens.consume_ws()
                    
                    # Record the correct answer.
                    if is_correct:
                        self.current_question.set_correct(Option(letter))
                    
                    # Read to end of line.
                    words = self.tokens.get_text_until(TokenType.EOL)
                    self.tokens.eat()
                    
                    # Possibly subsequent indented lines. They all have more than one leading space.
                    leading_ws = self.tokens.consume_ws()
                    while leading_ws > 1:
                        words += " " + self.tokens.get_text_until(TokenType.EOL)
                        self.tokens.eat()
                        leading_ws = self.tokens.consume_ws()
                    
                    # Push back the whitespace we just read
                    self.tokens.unget(leading_ws)
                    
                    self.current_question.add_option(Option(letter), words)
                    
                    # If we read all the options, look for the next question.
                    if letter == 'D':
                        # Sanity-check
                        if not self.current_question.is_valid():
                            self.fail(f"Invalid question:\n{self.current_question.formatted()}")

                        # End of file ahead?
                        consumed = self.tokens.consume_ws()
                        if self.tokens.peek().type == TokenType.EOF:
                            self.state = State.DONE
                        else:
                            self.tokens.unget(consumed)
                            self.state = State.GROUP_OR_QUESTION
                        
                case State.DONE:
                    "Done reading this file"
                    print(f"Read {len(self.questions)} questions from {self.file_path}.")
                    return
                        
                case _:
                    raise Exception(f"Unhandled state: {self.state}")


class QuizApp:

    def __init__(self):
        self.question_pool: QuestionPool = QuestionPool()
        self.subelement_question_map = {}
        self.question_weight_map = {}

    def load_questions(self, exam_level: str) -> None:
        cwd = os.getcwd()
        for filename in os.listdir(os.path.join(cwd, exam_level)):
            filepath = os.path.join(cwd, exam_level, filename)
            question_parser = QuestionParser(self.question_pool, filepath)
            question_parser.parse()

        for subelement in self.question_pool.subelements:
            self.subelement_question_map[subelement.name] = []

        for question in self.question_pool.questions:
            self.subelement_question_map[question.get_subelement_name()].append(question)
            self.question_weight_map[question.number] = 1

    def select_random(self, question_set: list[Question]) -> Question:
        return random.choices(
            population = question_set,
            weights = [self.question_weight_map[q.number] for q in question_set],
            k = 1
        )[0]

    def guess(self, question: Question, letter: Option) -> Option:
        index = self.question_pool.questions.index(question)
        if question.correct == letter:
            self.question_weight_map[index] = max(self.question_weight_map[question.number] / 3, 1)
        else:
            self.question_weight_map[index] = min(self.question_weight_map[question.number] * 3, len(self.question_pool) / 2)
        return question.correct

    def quiz(self, question_set: list[Question]) -> None:
        keystroke = ""
        while keystroke.lower() != "q":
            question = self.select_random(question_set)
            print("-----------------------------------------------------------")
            print(question.formatted())

            keystroke_valid = False
            while not keystroke_valid:
                keystroke = input("[a, b, c, d, q] > ")

                if keystroke.lower() == 'q':
                    print("Ok\n")
                    keystroke_valid = True

                elif keystroke.upper() in ['A', 'B', 'C', 'D']:
                    keystroke_valid = True
                    option = Option(keystroke.upper())
                    correct = self.guess(question, option)
                    if option == correct:
                        print("Yay\n")
                    else:
                        print(f"Alas. The correct answer was: {correct.value}\n")

    def main_menu(self) -> None:
        keystroke = ""
        while keystroke != "q":
            i = 1
            for subelement in self.question_pool.subelements:
                print(f"[{i}] {subelement.name}: {subelement.title}")
                i += 1
            print("[a] All")
            print("[q] Quit")

            keystroke_valid = False
            while not keystroke_valid:
                keystroke = input(f"[1 .. {len(self.question_pool.subelements)}, a, q] > ")

                if keystroke == "q":
                    keystroke_valid = True
                    print("Ok")

                if keystroke == "a":
                    keystroke_valid = True
                    self.quiz(self.question_pool.questions)

                if keystroke.isdigit() and 1 <= int(keystroke) <= len(self.question_pool.subelements):
                    keystroke_valid = True
                    subelement = self.question_pool.subelements[int(keystroke) - 1]
                    question_set = [q for q in self.question_pool.questions if q.get_subelement_name() == subelement.name]
                    self.quiz(question_set)

def main() -> int:
    questions = QuizApp()
    questions.load_questions('general')
    questions.main_menu()
    return 1


if __name__ == '__main__':
    sys.exit(main())

