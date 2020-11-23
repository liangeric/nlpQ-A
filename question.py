'''
Class that will process the question file and then return information about each question
'''

WHAT = "WHAT"
WHEN = "WHEN"
WHO = "WHO"
WHERE = "WHERE"
WHY = "WHY"
HOW = "HOW"
WHICH = "WHICH"
BINARY = "BINARY"


class Question:
    def __init__(self):
        # [string] raw question
        self.raw_question = None

        # [string] type of question i.e. WHO, BINARY, etc.
        self.question_type = None

        # [string] sentence without question_type i.e WHO, WHAT, etc.
        self.parsed_version = None
        self.sent_vector = None  # sent2vec on the parsed questions
        self.spacyDoc = None
        # list of top-k answer sentences, in the format of a spacy sent objects
        self.answers = [] 
        self.score = []

# DEPRECIATED
class QuestionProcess:
    def __init__(self):
        self.questions = None
        self.question_words = set([WHAT, WHEN, WHO, WHERE, WHY, HOW, WHICH])

    def get_question_class(self, raw_question):
        """[get the question class for a raw_question storing question_type, modified sentence, raw question]

        Args:
            raw_question ([string]): [the raw question from input]

        Returns:
            Question: Question class representation of raw_question
        """
        words = raw_question.split(" ")
        index = 0
        while index < len(words):
            question_type = words[index].upper()
            if question_type in self.question_words:
                index += 1
                break
            index += 1

        question_type = BINARY if question_type not in self.question_words else question_type
        return Question(raw_question, question_type, " ".join(words[index:]))

    def process_questions(self, file):
        """[process the question from a file]

        Args:
            file ([string]): [path to file containing list of questions separated by newline]

        Returns:
            [Question]: List of questions with class question
        """
        question_file = open(file, "r+")
        self.questions = question_file.read()
        self.questions = self.questions.split("\n")

        self.questions = [line.strip() for line in self.questions]

        return [self.get_question_class(question) for question in self.questions]


# sample code for demonstration purposes
if __name__ == "__main__":
    QP = QuestionProcess()
    questions = QP.process_questions("q.txt")
    for q in questions:
        print(q.raw_question, q.question_type, q.parsed_version)
