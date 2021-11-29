import nltk
import numpy as np
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# import and reading the corpous
f = open('chatbot.txt', 'r', errors='ignore')
raw_doc = f.read()
raw_doc = raw_doc.lower()
nltk.download('punkt')
nltk.download('wordnet')
sent_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.sent_tokenize(raw_doc)

# text preprocessing
lemmer = nltk.stem.WordNetLemmatizer()


def LemToken(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemToken(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# define greeting functions

GREET_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREET_RESPONSE = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


def greet(sentence):
    for word in sentence.split():
        if word.lower() in GREET_INPUTS:
            return random.choice(GREET_RESPONSE)


# Response Generations


def response(user_response):
    robot1_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=['ha', 'le', 'u', 'wa', 'english'])
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        robot1_response += "I am sorry! I don't understand you"
        return robot1_response
    else:
        robot1_response += sent_tokens[idx]
        return robot1_response


# Defining conversation start/end protocols

flag = True
print("BOT: My name is Jasmine. Let's have a conversation! Also, if you want to exit anytime, just type Bye!")

while flag:
    print("User: ", end="")
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("BOT: You are welcome...")
        else:
            if greet(user_response) is not None:
                print("BOT: " + greet(user_response))
            else:
                sent_tokens.append(user_response)
                word_tokens += nltk.wordpunct_tokenize(user_response)
                final_words = list(set(word_tokens))
                print("BOT: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)

    else:
        flag = False
        print("BOT: Goodbye! Take care <3 ")