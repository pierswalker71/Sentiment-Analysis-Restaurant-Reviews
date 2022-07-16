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

    from keras.models import Sequential
    from keras.layers import Dense,Dropout

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
    
    #==============================================================================
    # Functions 
    
    def lemmatization(text_list, en, stopwords):
        corpus = []
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
    
    # Create corpus us review text, removing stop words and other characters
    text_list = input_data['Review']
    corpus = lemmatization(text_list, en, stopwords)
    
    countvector = CountVectorizer()
    X = countvector.fit_transform(corpus).toarray()
    y = input_data['Liked'].values
    
    #countvector.get_feature_names_out()
    
    # Train classifier
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # MultinomialNB
    classifier = MultinomialNB(alpha=0.1)
    model_type = 'sklearn'
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
   
    # BernoulliNB
    #classifier = BernoulliNB(alpha=0.1)
    #model_type = 'sklearn'
    #classifier.fit(X_train, y_train)
    #y_pred = classifier.predict(X_test)
    
    #classifier = LogisticRegression()
    
    #def 

    confusion_matrix = confusion_matrix(y_test, y_pred)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    st.write('confusion_matrix')
    st.write(confusion_matrix)
    st.write(f'accuracy: {round(acc*100,2)} %')
    st.write(f'precision: {round(prec*100,2)} %')
    st.write(f'recall: {round(recall*100,2)} %')      
          
          
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
          

    st.header('Predictions')
    #new_comments = ['I liked the soup','Loved the beef','I hate waiting in this restaurant','staff were great']
    new_comments = st.text_input(label='new review', value='I liked the soup')
    text_spacy = lemmatization(new_comments, en, stopwords)
    
    st.write('key word compoents in your review')
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
    
    st.write('prediction')
    st.write(prediction)
    
    
    
    
    
    
    
    
    

if __name__ == '__main__':
    main()
