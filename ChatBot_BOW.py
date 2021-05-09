#importing libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

import os
#os.getcwd()
os.chdir("C:\\Users\\akadali\\Desktop\\Deep_NLP\\MLG_Capstone_ChatBot\\ChatBot_BOW_NN")

try:
    #In case of any new data added to the faq.json file, please delete "data.pickle" file from the 
    #directory before running the code. 
    with open("data.pickle", 'rb') as f:
        words, labels, train_x, train_y = pickle.load(f)
except:
    #Creating empty lists of words, classes and documents 
    words = []
    classes = []
    documents = []
    
    faq_file = open("faq.json").read()
    faq = json.loads(faq_file)
    
    for q in faq["faq"]:
        for pattern in q['patterns']:
            #tokenizing the documents to words
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            #creating documents with token lists and class
            documents.append((tokens, q['tag']))
            if q['tag'] not in classes:
                classes.append(q['tag'])
    #print(documents)
    #print(len(words))
    ignore_letters = ['!', '#', '%', '^', '&', '*', '?', '/']
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))
    
    #Printing number of documents, classes and lemmatized words
    print("\n")
    print(len(documents), " documents")
    print("\n")
    print(len(classes), " classes")
    print("\n")
    print(len(words), " unique lemmatized words")
    
    #Creating words and classes pickle files
    
    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))
    
    #Creating Training Dataset with bag of words
    training = []
    
    #Creating an empty list to save the one-hot encoded form (BAG OF WORDS) of output classes as target variable
    output_empty = [0]*len(classes)
    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
        for word in words:
            if word in pattern_words:
                bag.append(1)
            else:
                bag.append(0)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])
    
    #Shuffling the training dataset
    random.shuffle(training)
    training = np.array(training)
    #Creating dependent and independent lists 
    train_x = list(training[:,0])
    train_y = list(training[:,1])
    
    with open("data.pickle", "wb") as f:
        pickle.dump((words, classes, train_x, train_y), f)
    print("Training data is ready")

try:
    #In case of any new data added to the "faq.json" file, please delete "chatbo_mode.h5" file from the directory 
    model.load('chatbot_model.h5')

except:
    #Create model - 3 layers. 
    #First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
    # equal to number of intents to predict output intent with softmax
    
    model = tf.keras.Sequential([
    							 tf.keras.layers.Dense(128, input_shape =(len(train_x[0]), ) , activation = 'relu'),
    							 tf.keras.layers.Dropout(0.5),
    							 tf.keras.layers.Dense(64, activation = 'relu'),
    							 tf.keras.layers.Dropout(0.5),
    							 tf.keras.layers.Dense(len(train_y[0]), activation = 'softmax')
    							 ])
    
    # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
    sgd = tf.keras.optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    #fitting and saving the model
    hist = model.fit(np.array(train_x), np.array(train_y), epochs = 200, batch_size = 5, verbose = 1)
    model.save('chatbot_model.h5', hist)

def bag_of_words(s, words):
    bag = [0]*len(words)
    s_words = nltk.word_tokenize(s)
    s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words if word not in ignore_letters]
    for word in s_words:
        for i,w in enumerate(words):
            if w == word:
                bag[i] = 1
    return np.array(bag)
"""
#Sample Testing
test_q = "How long will talent referrals stay on file at Deloitte?"
q_bag = bag_of_words(test_q, words)      
print(q_bag)
q_bag.shape
#Converting 
q_bag = np.array([q_bag])
q_bag.shape
#q_bag.reshape(184,1)
result = model.predict(q_bag)
print(result)
result_index = np.argmax(result)
print(result_index)
print(result[0][result_index])
print(classes[result_index])
label = classes[result_index]
#print(documents)
faq_file = open("faq.json").read()
faq = json.loads(faq_file)
for q in faq['faq']:
    if q['tag'] == label:
      responses = q['responses']
print(responses)
"""
#Create an empty list to collect the user queries for further review
userInputQueue = []
botOutputQueue = []
def chat():
    print("Start talking with the bot (type 'quit' to stop)")
    while True:
        user_input = input("You: ")
        userInputQueue.append(user_input)
        if user_input.lower() == 'quit':
            print("Do you want to save your conversation?? [Y/N]\n")
            input_file_flag = input("Your response: ")
            if input_file_flag.lower() == 'y':
                conversation = open("user_inputs.txt", "w")
                for que in userInputQueue:
                    for ans in botOutputQueue:
                        conversation.write("User:")
                        conversation.write(que)
                        conversation.write("\n")
                        conversation.write("Bot:")
                        conversation.write(ans)
                        conversation.write("\n")
                conversation.close()
                print("Your conversation has been saved. Thank you")
            break
        bow = np.array([bag_of_words(user_input, words)])
        results = model.predict(bow)
        #print("Results\n",results)
        result_index = np.argmax(results) 
        #print("Result Index\n",result_index)
        label = classes[result_index]
        
        for q in faq['faq']:
            if q['tag'] == label:
                response = q['responses']
                botOutputQueue.append(response)
                print("Bot:", response)
            #default bot response
            #print("I didn't get that, try again.")
#classes[10]
#Let's call Chat function to test the bot
chat()

#Create a text file of user inputs to train the bot in future

    
    

    
    
    
    
    
    