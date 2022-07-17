import streamlit as st

def main():
    
    #==============================================================================
    # Imports
    
    import pandas as pd
    import re 
    
    import spacy
    en = spacy.load("en_core_web_sm")
    stopwords = en.Defaults.stop_words

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.linear_model import LogisticRegression
    
    from sklearn.svm import SVC

    from sklearn.ensemble import RandomForestClassifier


    from sklearn.neural_network import MLPClassifier

    from keras.models import Sequential
    from keras.layers import Dense,Dropout

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score

    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    
    
    #==============================================================================
    # Settings
    st.set_page_config(page_title = 'Sentiment Analysis') 
    
    # Title
    st.title('Sentiment Analysis - Restaurant Reviews')
    st.write('Piers Walker 2022. https://github.com/pierswalker71')
    st.write('This tool is capable of establishing whether restaurant reviews are positive or negative.')
    st.write('It does this by training a machine learning model on an open data source with prelabled reviews.')
    st.write('The machine learning model first extracts the key components of each review text, dropping irrelevant words, and simplifying each word to its root meaning.')
    st.write('The classification model then learns which groups of words correspond to either positive or negative sentiment.')
    st.write('Once trained, the model is capable of making predictions on brand new review text.')
    st.write('Try it out yourself by providing your own culinary review.')


    #==============================================================================
    # Functions 
    
    def lemmatization(text_list, en, stopwords):
        corpus = []
        if isinstance(text_list, str):
            #new_list = []
            #text_list = new_list.append(text_list)
            text_list = [text_list]
        for txt in text_list:
            new_text = re.sub(pattern='[^a-zA-z]', repl=' ', string=txt)
            new_text = new_text.lower()
            new_text = en(new_text)
            new_text = [token.lemma_ for token in new_text if str(token) not in stopwords]
            new_text = ' '.join(new_text)
            corpus.append(new_text)
        return corpus 

    #==============================================================================
    
    # Load data
    st.header('Load review data')
    input_data = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t')

    st.dataframe(input_data)
    
    # 
    
    # Create corpus of review text, removing stop words and other characters
    text_list = input_data['Review']
    corpus = lemmatization(text_list, en, stopwords)
    
    countvector = CountVectorizer()
    X = countvector.fit_transform(corpus).toarray()
    y = input_data['Liked'].values
    
    #countvector.get_feature_names_out()
    
    # Train classifier
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    st.header('Select and train model')     

    # select model 

    #classifier = LogisticRegression()
    classifier = MultinomialNB(alpha=0.1)
    #classifier = BernoulliNB(alpha=0.1)  
    #classifier = SVC() #kernel='linear
    #classifier = MLPClassifier()
   # classifier = RandomForestClassifier()

    model_type = 'sklearn'
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    
    
    
    # Keras neural network 

    input_dim = X.shape[1]

    #classifier = Sequential()
    #model_type = 'keras'
    #classifier.add(Dense(2000, input_dim=input_dim))
    #classifier.add(Dropout(0.5))
    #classifier.add(Dense(2000))
    #classifier.add(Dropout(0.5))
    #classifier.add(Dense(1, activation="sigmoid"))
    #classifier.compile(loss='binary_crossentropy', metrics='accuracy')
    #classifier.fit(X_train, y_train, epochs=20,)    
   
    #continuous_values = classifier.predict(X_test)

    # Convert to binary
    #binary_values = []
    #for i in continuous_values:
    #    if (i[0]<0.5):
    #        binary_values.append(0)
    #    else:
    #        binary_values.append(1)
    #y_pred = binary_values

 #def 

    

    st.header('Model performance')
    confusion_matrix = confusion_matrix(y_test, y_pred)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    st.write('confusion_matrix')
    st.write(confusion_matrix)
    st.write(f'accuracy: {round(acc*100,2)} %')
    st.write(f'precision: {round(prec*100,2)} %')
    st.write(f'recall: {round(recall*100,2)} %')      
          
    st.write(f'roc_auc_score: {round(auc*100,2)}')
    st.write(f'f1 score: {round(f1*100,2)}')       
      
          
   
          

    st.header('Predictions')
    #st.write('Provide a new restaurant review for the model to analyse.')   #new_comments = ['I liked the soup','I hate waiting in this restaurant','Loved the beef','staff were great']
    new_comments = st.text_input(label='Provide a new restaurant review for the model to analyse.', value='I liked the soup')
    text_spacy = lemmatization(new_comments, en, stopwords)
    
    st.write('key word components in your review')
    st.write(text_spacy)

    # Make prediction
    prediction = classifier.predict(countvector.transform(text_spacy))
   

    #if model_type == 'keras':
    #    continuous_values = prediction
    #    # Convert to binary
    #    binary_values = []
    #    for i in continuous_values:
    #        if (i[0]<0.5):
    #            binary_values.append(0)
    #        else:
    #            binary_values.append(1)
    #    prediction = binary_values
    
    st.write('Here is my prediction')
    st.write(prediction)
    
    
    
    
    
    
    
    
    

if __name__ == '__main__':
    main()
