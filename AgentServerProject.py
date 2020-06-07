import socket
import threading

HEADER = 64
PORT = 5050
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "disconnect"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

#################################################
##########Load a whole bunch of stuff############
#################################################


import spacy
from spacy import displacy
from spacy.lang.en import English
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import pickle
from collections import Counter
spacy.load('/Users/donaldcrowley/PycharmProjects/untitled2/env/lib/python3.7/site-packages/en_core_web_sm/en_core_web_sm-2.2.0')
#import en_core_web_sm
#pip install textblob
from textblob import TextBlob
#pip install transformers
import torch
from transformers import BertForQuestionAnswering
model2 = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
from transformers import BertTokenizer
tokenizer2 = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
import pandas as pd
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
nlp = spacy.load('/Users/donaldcrowley/PycharmProjects/untitled2/env/lib/python3.7/site-packages/en_core_web_sm/en_core_web_sm-2.2.0')
from nltk.corpus import brown
nltk.download("punkt")
from nltk.corpus import PlaintextCorpusReader

#################################################
###############End of Loading Zone###############
#################################################

##################################################
################Setup for Neural Networks#########
##################################################
# Use a GPU if you have one available (Runtime -> Change runtime type -> GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds for reproducibility
random.seed(26)
np.random.seed(26)
torch.manual_seed(26)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
model.to(device) # Send the model to the GPU if we have one

learning_rate = 1e-5
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
def encode_data(tokenizer, questions, passages, max_length):
    """Encode the question/passage pairs into features than can be fed to the model."""
    input_ids = []
    attention_masks = []

    for question, passage in zip(questions, passages):
        encoded_data = tokenizer.encode_plus(question, passage, max_length=max_length, pad_to_max_length=True, truncation_strategy="longest_first")
        encoded_pair = encoded_data["input_ids"]
        attention_mask = encoded_data["attention_mask"]

        input_ids.append(encoded_pair)
        attention_masks.append(attention_mask)

    return np.array(input_ids), np.array(attention_masks)

# Loading data
#train_data_df = pd.read_json("/content/train.jsonl", lines=True, orient='records')
#dev_data_df = pd.read_json("/content/dev.jsonl", lines=True, orient="records")
train_data_df = pd.read_json("train.jsonl", lines=True, orient='records')
dev_data_df = pd.read_json("dev.jsonl", lines=True, orient="records")

passages_train = train_data_df.passage.values
questions_train = train_data_df.question.values
answers_train = train_data_df.answer.values.astype(int)

passages_dev = dev_data_df.passage.values
questions_dev = dev_data_df.question.values
answers_dev = dev_data_df.answer.values.astype(int)

# Encoding data
max_seq_length = 256
input_ids_train, attention_masks_train = encode_data(tokenizer, questions_train, passages_train, max_seq_length)
input_ids_dev, attention_masks_dev = encode_data(tokenizer, questions_dev, passages_dev, max_seq_length)

train_features = (input_ids_train, attention_masks_train, answers_train)
dev_features = (input_ids_dev, attention_masks_dev, answers_dev)


batch_size = 32

train_features_tensors = [torch.tensor(feature, dtype=torch.long) for feature in train_features]
dev_features_tensors = [torch.tensor(feature, dtype=torch.long) for feature in dev_features]

train_dataset = TensorDataset(*train_features_tensors)
dev_dataset = TensorDataset(*dev_features_tensors)

train_sampler = RandomSampler(train_dataset)
dev_sampler = SequentialSampler(dev_dataset)

train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=batch_size)

########################################################
########################this should be 5################
########################################################
########################################################
epochs = 1
grad_acc_steps = 1
train_loss_values = []
dev_acc_values = []

for _ in tqdm(range(epochs), desc="Epoch"):

    # Training
    epoch_train_loss = 0  # Cumulative loss
    model.train()
    model.zero_grad()

    for step, batch in enumerate(train_dataloader):

        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)
        labels = batch[2].to(device)

        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks, labels=labels)

        loss = outputs[0]
        loss = loss / grad_acc_steps
        epoch_train_loss += loss.item()

        loss.backward()

        if (step + 1) % grad_acc_steps == 0:  # Gradient accumulation is over
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clipping gradients
            optimizer.step()
            model.zero_grad()

    epoch_train_loss = epoch_train_loss / len(train_dataloader)
    train_loss_values.append(epoch_train_loss)

    # Evaluation
    epoch_dev_accuracy = 0  # Cumulative accuracy
    model.eval()

    for batch in dev_dataloader:
        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)
        labels = batch[2]

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()

        predictions = np.argmax(logits, axis=1).flatten()
        labels = labels.numpy().flatten()

        epoch_dev_accuracy += np.sum(predictions == labels) / len(labels)

    epoch_dev_accuracy = epoch_dev_accuracy / len(dev_dataloader)
    dev_acc_values.append(epoch_dev_accuracy)

#save this model as the ynmodel
ynmodel = model

##################################################
#############CLASSES AND FUNCTIONS################
##################################################

#create a class 'Person'
#this will enable the AI to learn about the user's opinion on people in the text

#load in our list of people, otherwise, create a blank list
def load_people():
  try:
    peoplelist = pickle.load(open("peoplelist.pkl", "rb"))
  except:
    peoplelist = []

class Person:
    def __init__(self, name, useropinion, opinionstrength, descriptions):
        self.name = name
        self.useropinion = useropinion
        self.opinionstrength = opinionstrength
        #create a list of descriptions of this new person
        self.descriptions = descriptions

def name_finder(story):
    nlp = spacy.load('/Users/donaldcrowley/PycharmProjects/untitled2/env/lib/python3.7/site-packages/en_core_web_sm/en_core_web_sm-2.2.0')




    doc = nlp(story)
    sentences = [x for x in doc.sents]
    #return the names in the story
    names = (dict([(str(x), x.label_) for x in nlp(str(sentences[0:])).ents]))
    return(set(names))



def build_opinions(names, peoplelist):
    for i in names:
        if i not in peoplelist:
            #see what the user thinks about this person who is new to the agent
            #print(conn1)
            #opinionbuild = input("what do you think about " + i + "? If this is not a name please type: not a name\n")

            #####
            conn1.send(("what do you think about " + i + "? If this is not a name please type: not a name\n").encode('utf-8'))
            msg_length = conn1.recv(HEADER).decode(FORMAT)
            msg = conn1.recv(int(msg_length)).decode(FORMAT)
            opinionbuild = msg
            print(opinionbuild)
            #########

            if opinionbuild != "not a name":

              text = nltk.word_tokenize(opinionbuild)
              tagged = nltk.pos_tag(text)
              emplist = []
              for j in tagged:
                  if j[1] == 'JJ':
                      emplist.append(j[0])
              peoplelist.append(Person(i, TextBlob(opinionbuild).sentiment[0], TextBlob(opinionbuild).sentiment[1], emplist))
        else:
            print("you described " + i +" as ")

            #####
            desc = "you described " + i +" as "
            conn1.send((desc).encode('utf-8'))

            #####
            for word in peoplelist[0].descriptions:
                print(word)
                conn1.send(word)
    return(peoplelist)


def save_object(obj, filename):
    #overwrite the existing database
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
#uses the save_object file to save the file if user requests it
def savepeople():
    #saveYN = input("Would you like to save your people opinions from this story? \n")
    conn1.send(("Would you like to save your people opinions from this story?").encode('utf-8'))
    msg_length = conn1.recv(HEADER).decode(FORMAT)
    msg = conn1.recv(int(msg_length)).decode(FORMAT)
    saveYN = msg
    if saveYN == "y" or saveYN == "yes" or saveYN == "Yes":
        save_object(peoplelist, 'peoplelist.pkl')


def answer_question(question, answer_text):
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer2.encode(question, answer_text)
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer2.sep_token_id)
    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1
    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a
    # Construct the list of 0s and 1s.
    segment_ids = [0] * num_seg_a + [1] * num_seg_b
    # There should be a segment_id for every input token.
    #the assert function checks that something is true, otherwise it raises the error message provided
    assert len(segment_ids) == len(input_ids), "missing a segment id for the input token"


    # Run our example question through the model.
    start_scores, end_scores = model2(torch.tensor([input_ids]),  # The tokens representing our input text.
                                     token_type_ids=torch.tensor(
                                         [segment_ids]))  # The segment IDs to differentiate question from answer_text

    # the tokens with the highest start and end scores emcompass the likeliest start and end
    #to the answer of the question
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    # Get the string versions of the input tokens.
    tokens = tokenizer2.convert_ids_to_tokens(input_ids)
    # Start with the first token.
    answer = tokens[answer_start]
    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        # If it's a subword token(denoted by ## in BERT), then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]
    #print('Answer: "' + answer + '"')
    #return(str(answer))
    return (answer)


def predictyn(question, passage):
    sequence = tokenizer.encode_plus(passage, question, return_tensors="pt")['input_ids'].to(device)

    logits = ynmodel(sequence)[0]
    probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
    proba_yes = round(probabilities[1], 2)
    proba_no = round(probabilities[0], 2)

    # Our prediction wasn't working super well above, but I think
    # we can just say if the predicted probability of no is higher
    # there is a good chance the answer is no, even if the absolute probability
    # of 'yes' is high
    if proba_yes > proba_no:
        return ("yes")
    else:
        return ("no")

def goal_query(person, story):
  ques = "what was " + str(person.name) + " supposed to do?"
  answer1 = answer_question(ques, str(story))
  question_affirmative = "did " + str(person.name) + " "+ str(answer1) + "?"
  question_negative = "did " + str(person.name) + " not "+ str(answer1) + "?"
  #used the ratio to answer a yes/no question about whether that goal was achieved
  #going through sentence by sentence and getting the max seemed to be a more effective way of doing this
  #so the 'levensh' function does that

  if predictyn(question_affirmative, answer_question(question_affirmative, story)) > predictyn(question_negative, answer_question(question_negative, story)):
    #print(person.name + " achieved the goal of " + str(answer1) + "!\n.  You are happy for " + person.name + " for achieving a goal.")
    emotion = "happy-for"
    response = person.name + " achieved the goal of " + str(answer1) + "!\n.  You are happy for " + person.name + " for achieving a goal."
    return(emotion, response)
  else:
    #print(person.name + " did not achieve the goal of " + "\'" + str(answer1) + "\'" +"\n"+ "  You are sorry for " + person.name + " for not achieving a goal.")
    emotion = "sorry-for"
    response = person.name + " did not achieve the goal of " + "\'" + str(answer1)  + "\'"+ "\n" + "  You are sorry for " + person.name + " for not achieving a goal."
    return(emotion, response)

class user_goal:
    def __init__(self, goal, exclusive):
        #what is the goal
        self.goal = goal
        #is it mutually exclusive (jealousy)?  value = 1
        #is it a desired non-exclusive(envy)?  value = 2
        #or neither.  A goal that can be shared by user and other person?  value = 0
        self.exclusive = exclusive

def user_goal_query(user_goals, story):
    #check for each goal against the story and see if it was realized or not
    for goal in user_goals:
        questionY = "did I" + goal + "?"
        questionN = "did I not" + goal + "?"
        if predictyn(questionY, story) > predictyn(questionN, story):
            return (goal)


def user_query(story, emotion):
    #correctYN = input("was the predicted emotion correct? y/n \n")
    conn1.send(("was the predicted emotion correct? y/n").encode('utf-8'))
    msg_length = conn1.recv(HEADER).decode(FORMAT)
    msg = conn1.recv(int(msg_length)).decode(FORMAT)
    correctYN = msg
    if correctYN == "n":
        #actual_emotion = input("What was your actual emotional response? \n")
        conn1.send(("What was your actual emotional response?").encode('utf-8'))
        msg_length = conn1.recv(HEADER).decode(FORMAT)
        msg = conn1.recv(int(msg_length)).decode(FORMAT)
        actual_emotion = msg
        #altQuestion = input("what question could I have asked about this text to gauge my emotional reaction?  type 'n' if not. \n")
        conn1.send(("what question could I have asked about this text to gauge my emotional reaction?  type 'n' if not.").encode('utf-8'))
        msg_length = conn1.recv(HEADER).decode(FORMAT)
        msg = conn1.recv(int(msg_length)).decode(FORMAT)
        altQuestion = msg
        if altQuestion == 'n':
            #usergoal = input("if you had a goal in this story, what was it? \n")
            conn1.send(("if you had a goal in this story, what was it?").encode('utf-8'))
            msg_length = conn1.recv(HEADER).decode(FORMAT)
            msg = conn1.recv(int(msg_length)).decode(FORMAT)
            usergoal = msg
            if usergoal:
                goalfeatures = input(
                    "is this a mutually exclusive goal?  enter 1 \nis this a non-exclusive goal?  enter 2\nOr neither?  enter 0\n")
                conn1.send(("is this a mutually exclusive goal?  enter 1 \nis this a non-exclusive goal?  enter 2\nOr neither?  enter 0").encode('utf-8'))
                msg_length = conn1.recv(HEADER).decode(FORMAT)
                msg = conn1.recv(int(msg_length)).decode(FORMAT)
                goalfeatures = msg
            return (actual_emotion, usergoal, goalfeatures, 0)
        else:
            #usergoal = input("if you had a goal in this story, what was it? \n")
            conn1.send(("if you had a goal in this story, what was it? ").encode('utf-8'))
            msg_length = conn1.recv(HEADER).decode(FORMAT)
            msg = conn1.recv(int(msg_length)).decode(FORMAT)
            usergoal = msg
        return (actual_emotion, 0, 0, altQuestion)
    else:
        #print("great, you and I are on the same page!\n")
        conn1.send("great, you and i are on the same page!")
        return (0, 0, 0, 0)




##################################################
##################AI AGENT########################
##################################################

user_goal_list = []
memories_list = []
question_list = []


def AI_Agent(story, opinionlist, user_goal_list):
    # Emotion is what we want to predict
    Emotion = 0

    # ask user whether to load in saved people:
    #loadyn = input("Would you like to load in your saved list of people?(y/n)\n")
    conn1.send(("Would you like to load in your saved list of people?(y/n) ").encode('utf-8'))
    msg_length = conn1.recv(HEADER).decode(FORMAT)
    msg = conn1.recv(int(msg_length)).decode(FORMAT)
    loadyn = msg
    if loadyn.lower() == "y" or loadyn.lower() == "yes":
        load_people()

        ######################################################################
    ###Start off looking for emotions found within 'fortunes of others'###
    ######################################################################

    # identify the names in the story
    # and build opinions of each person
    opinions = build_opinions(name_finder(story), opinionlist)
    #print("\n Here is our story that provokes an emotion.  Let's see if we are both on the same page! \n")
    conn1.send(("Here is our story that provokes an emotion.  Let's see if we are both on the same page! \n ").encode('utf-8'))

    #print(story)
    conn1.send((story).encode('utf-8'))
    # each name in opinions is an object that contains:
    # 1 name
    # 2 user opinion
    # 3 the strength of that opinion
    # 4 any adjectives used to describe the person

    # check if a person in the story is a friend, neutral, or a rival
    # this section will test for Fortunes-of-Other's emotions
    for person in set(opinions):

        # we'll use a sentiment analysis of .5 as the cutoff for friend, but this can
        # be changed

        ################
        #####friend#####
        ################
        if person.useropinion > .5:

            # first try to determine if a goal has been achieved by the friend
            # if the goal has been achieved but the user has resentment about not
            # also achieving the goal, the user could either be jealous or envious
            emotion_predict, response = goal_query(person, story)
            #print(response)
            conn1.send((response).encode('utf-8'))
            if emotion_predict == "happy_for":

                # now check if the user wanted to obtain the same goal
                useGoal = user_goal_query(user_goals, story)
                # if so, are we envious or jealous?
                if useGoal == True:
                    if useGoal.exclusive == 1:
                        emotion_predict = "jealous of"
                        # return(emotion_predict)
                    elif useGoal.exclusive == 2:
                        emotion_predict = "envious of"
                        # return(emotion_predict)
                else:
                    #print(response)
                    conn1.send((response).encode('utf-8'))

            if emotion_predict == "sorry for":
                #print(response)
                conn1.send((response).encode('utf-8'))

        ###############
        #####rival#####
        ###############

        if person.useropinion < (-.5):
            # in this case we are gauging the response to a person
            # that the user doesn't especially like
            emotion_predict, response = goal_query(person, story)
            if emotion_predict == "happy_for":
                # so this person achieved their goal, but since the
                # user doesn't like them, we'll flip the opinion to negative
                emotion_predict = "resentment"
            # now what if the rival didn't achieve their goal:
            else:
                emotion_predict = 'gloating'

                print(response.replace("sorry for", "gloating at"))
                conn1.send((response.replace("sorry for", "gloating at")).encode('utf-8'))

    # if
    emotion_actual, usergoal, newfeature, Alt_question = user_query(story, emotion_predict)
    # if the user has given a new goal, we add it to the list of user's goals
    if usergoal:
        newgoal1 = user_goal(usergoal, newfeature)
        user_goal_list.append(newgoal)
    # if our actual emotion is our predicted emotion
    if emotion_actual == 0:
        emotion_actual = emotion_predict

    # should we save the people list?
    #pepsav = input("would you like to save our opinions on people? y/n\n")


    if pepsav.lower == 'y':
        savepeople()
    # should we save the actual emotion, the user goal and the question asked?

    # would you like to keep going?
    #go = input("would you like to continue?  y/n \n")
    conn1.send(("would you like to continue?  y/n ").encode('utf-8'))
    msg_length = conn1.recv(HEADER).decode(FORMAT)
    msg = conn1.recv(int(msg_length)).decode(FORMAT)
    go = msg

    if go.lower == 'n':
        #print(emotion_predict)
        conn1.send(("emotion_predict").encode('utf-8'))
        #print(emotion_actual)
        conn1.send(("emotion_actual ").encode('utf-8'))

        return (story, name_finder(story), emotion_predict, emotion_actual)


# now get feedback from the user on if this interpretation was correct
# user_query_interp(story, person, Emotion)





###################################################
#############SERVER SETUP##########################
###################################################

import socket
import threading

HEADER = 64
PORT = 5050
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"

# this function is to handle the socket connection
# take the incoming message, and produce an outgoing reply
def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    global conn1
    conn1 = conn

    while connected:
        global msg_length
        # receiving the message from the client and turning into a string
        msg_length = conn.recv(HEADER).decode(FORMAT)
        print(msg_length)
        if msg_length:
            msg_length = int(msg_length)
            global msg
            msg = conn.recv(msg_length).decode(FORMAT)
            # close the connection if user types 'disconnect'
            if msg == DISCONNECT_MESSAGE:
                connected = False
        '''global msg
        msg = conn.recv(message).decode(FORMAT)'''

        keepgoing = True

        while keepgoing == True:
            #story = input("Would you please give me a story? \n")
            conn1.send(("Would you please give me a story?").encode('utf-8'))
            msg_length = conn1.recv(HEADER).decode(FORMAT)
            msg = conn1.recv(int(msg_length)).decode(FORMAT)
            story = msg
            try:
                story1, names, predictedEmotion, actualEmotion = AI_Agent(story, [], [])
            except:
                break


        '''if reply:
            conn.send((reply).encode('utf-8'))'''

    # close connection
    conn.close()


# this function starts up the server and provides input into the handle_client function
def start():
    server.listen()
    print(f"[LISTENING] Server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")


print("[STARTING] server is starting...")






start()

