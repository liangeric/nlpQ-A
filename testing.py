from answer import *

def testAnswer():
    article = open("a1.txt", "r", encoding="UTF-8").read()
    questions = open("q.txt", "r", encoding="UTF-8").read()
    correctAnswers = open("a1_correct.txt", encoding="UTF-8").read().splitlines()

    answer = Answer(article, questions)
    answer.preprocess()
    qsObjLst = answer.similarity(distFunc=jaccardSim, k = 5)

    scoring = []
    noAns = 0
    for qInd in range(len(qsObjLst)):
        qObj = qsObjLst[qInd]
        # Get question
        orgQuestion = qObj.raw_question
        print("Question: {}".format(qObj.raw_question))

        # Scoring answer
        answerScore = None
        actualAnswerSent = correctAnswers[qInd]

        for i in range(len(qObj.answers)):
            # Get answer
            orgAnswer = qObj.answers[i]
            print("Answer {}: {}".format(i, orgAnswer))

            print("Found Answer:")
            print(answer.answerQuestion(orgQuestion, orgAnswer))
            print("\n")
            if orgAnswer.text == actualAnswerSent:
                answerScore = i
        
        if answerScore is None:
            scoring.append("No Answer")
            noAns += 1
        else:
            scoring.append(str(answerScore))

        print("\n")
    return scoring, len(scoring) - noAns # list of index where answer is found, if any (No Answer o/w) and the number correct



if __name__ == "__main__":
    print(testAnswer())
    