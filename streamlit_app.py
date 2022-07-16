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
    
    url_data = (r'https://github.com/pierswalker71/Sentiment-Analysis-Restaurant-Reviews/blob/main/Restaurant_Reviews.tsv')
    input_data = pd.read_csv(url_data,delimiter='\t')
    
    st.dataframe(input_data)
    
    
    
    
    
    
    
    
    
    
    
    
    
    

if __name__ == '__main__':
    main()
