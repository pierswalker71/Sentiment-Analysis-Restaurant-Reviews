import streamlit as st

def main():
    
    #==============================================================================
    # Imports
    
    import pandas as pd
    import re 
    
    import spacy
    en = spacy.load("en_core_web_sm")
    #en = spacy.load("en_core_web_md")
    stopwords = en.Defaults.stop_words

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.linear_model import LogisticRegression

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
    
    # Set load data flag and blank model type. These are used to execute data load and model training a minimal number of times
    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False
    if 'model_type' not in st.session_state:
        st.session_state['model_type'] = ''
    if 'classifier' not in st.session_state:     
        st.session_state['classifier'] = ''
    
    # Title
    st.title('Sentiment Analysis - Restaurant Reviews')
    st.write('Piers Walker 2022. https://github.com/pierswalker71')
    st.write('This tool is capable of establishing whether restaurant reviews are positive or negative.')
    st.write('It does this by training a machine learning model on an open data source with prelabled reviews (https://www.kaggle.com/datasets/d4rklucif3r/restaurant-reviews).')
    st.write('The machine learning model first extracts the key components of each review text, dropping irrelevant words, and simplifying each word to its root meaning.')
    st.write('The classification model then learns which groups of words correspond to either positive or negative sentiment.')
    st.write('Once trained, the model is capable of making predictions on brand new review text.')
    st.write('Try it out yourself by providing your own culinary review.')
    st.write('Note: Different machine learning models may be selected from the side bar. Their training performance is presented underneath.')


    #==============================================================================
    # Functions 
    
    def lemmatization(text_list, en, stopwords):
        # Creates a corpus of text from a list of sentences
        
        corpus = []
        if isinstance(text_list, str):
            text_list = [text_list]
        for txt in text_list:
            new_text = re.sub(pattern='[^a-zA-z]', repl=' ', string=txt) # Removes non-alpha characters
            new_text = new_text.lower()
            new_text = en(new_text)
            new_text = [token.lemma_ for token in new_text if str(token) not in stopwords]
            new_text = ' '.join(new_text)
            corpus.append(new_text)
        return corpus 
    #-----------------------------------------------------------------------------
    def threshold_to_binary(continuous_values):
        # Converts list of coeficients to binary, 0 or 1, based on threshold of 0.5 
        
        binary_values = []
        for i in continuous_values:
            if i[0] < 0.5:
                binary_values.append(0)
            else:
                binary_values.append(1)
        return binary_values
    #-----------------------------------------------------------------------------    
    
    def build_classifier(model_type='Logistic Regression', keras_input_dimensions=1000):
        # Returns a classified based on the desired type
        # Input: model_type(str): Selects model. One of 'Logistic Regression', 'Bernoulli Naive Bayes', 'Neural Network'
        # Input: keras_input_dimensions(int): Only required if model_type=='Neural Network'
        
        if model_type == 'Logistic Regression':
            classifier = LogisticRegression()      
        elif model_type == 'Naive Bayes':
            classifier = MultinomialNB(alpha=0.1)
        elif model_type == 'Bernoulli Naive Bayes':
            classifier = BernoulliNB(alpha=0.1)                
        elif model_type == 'Neural Network':
            input_dim = keras_input_dimensions
            classifier = Sequential()
            classifier.add(Dense(2000, input_dim=input_dim))
            classifier.add(Dropout(0.5))
            classifier.add(Dense(2000))
            classifier.add(Dropout(0.5))
            classifier.add(Dense(1, activation="sigmoid"))
            classifier.compile(loss='binary_crossentropy', metrics='accuracy')   
            
        return classifier   
    
    #==============================================================================

    if st.session_state['data_loaded'] == False:
        # Load data
        st.header('Training data')
        input_data = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t')

        with st.expander('Review data'):
            st.dataframe(input_data)
            pos = len(input_data[input_data['Liked']==1].index)
            neg = len(input_data[input_data['Liked']==0].index)
            st.write(f'Number of positive reviews = {pos} ({round(pos*100/(pos+neg),2)}%)')
            st.write(f'Number of negative reviews = {neg} ({round(neg*100/(pos+neg),2)}%)')
     
        # Process text data
        # Create corpus of review text, removing stop words and other characters
        text_list = input_data['Review']
        corpus = lemmatization(text_list, en, stopwords)
    
        countvector = CountVectorizer()
        X = countvector.fit_transform(corpus).toarray()
        y = input_data['Liked'].values
    
        # Train test split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        
        st.session_state['training_flag'] = True
    
    #==============================================================================
    # User selection of model type model 
    with st.sidebar:
        st.header('Select model') 
        model_type = st.selectbox('Select model type', ['Logistic Regression','Naive Bayes', 'Bernoulli Naive Bayes','Neural Network'])
        
    #==============================================================================    
    if st.session_state['model_type'] != model_type:
        # Build training model of selected type
        classifier_training = build_classifier(model_type=model_type, keras_input_dimensions=X.shape[1])

        if model_type == 'Neural Network': # Keras neural network        
            classifier_training.fit(X_train, y_train, epochs=20,)    
            continuous_values = classifier_training.predict(X_test)
            y_pred = threshold_to_binary(continuous_values)
        else:
            classifier_training.fit(X_train, y_train)
            y_pred = classifier_training.predict(X_test)     

        #-----------------------------------------------------------------------------
        # Assess training results
        confusion_matrix = confusion_matrix(y_test, y_pred)
    
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        with st.sidebar:
            st.header('Model training performance')
            st.write('confusion_matrix')
            st.write(confusion_matrix)
            st.write(f'Accuracy: {round(acc*100,2)} %')
            st.write(f'Precision: {round(prec*100,2)} %')
            st.write(f'Recall: {round(recall*100,2)} %')      
          
            st.write(f'Roc auc score: {round(auc*100,2)}')
            st.write(f'f1 score: {round(f1*100,2)}')     
        
        # Update session state value with most recently trained model type
        st.session_state['model_type'] = model_type
    
        #==============================================================================
        # Rebuild classifier and train on whole dataset
        classifier = build_classifier(model_type=model_type, keras_input_dimensions=X.shape[1])
        # Add any optimal hyperparameters here
    
        if model_type == 'Neural Network':
            classifier.fit(X, y, epochs=20,)    
        else:
            classifier.fit(X, y)   
        # Update session state value with most recently trained model 
        st.session_state['classifier'] = classifier      
    
    #-----------------------------------------------------------------------------
    # Make prediction using user entered review text
    
    st.header('Enter new restuarant review')
    new_comments = st.text_input(label='Provide a new restaurant review for the model to analyse.', value='the soup was delightful')
    text_spacy = lemmatization(new_comments, en, stopwords)
         
    # Make prediction
    prediction = st.session_state['classifier'].predict(countvector.transform(text_spacy))
    
    # Convert continuous vaues to binary if required
    #if model_type == 'Neural Network':
    #    continuous_values = prediction
        
    #    binary_values = []
    #    for i in continuous_values:
    #        if i[0] < 0.5:
    #            binary_values.append(0)
    #        else:
    #            binary_values.append(1)
    #    prediction = binary_values
        
    st.header('Evaluation')    
    st.write(f'key word components found in your review: [{text_spacy[0]}]') 
    st.write('\nMy prediction:')
    #st.write(continuous_values)
    #st.write(binary_values)
    if prediction >0.5:
        st.markdown('**I think this is a positive review comment**')
    else:
        st.markdown('**I think this is a negative review comment**')
    


if __name__ == '__main__':
    main()
