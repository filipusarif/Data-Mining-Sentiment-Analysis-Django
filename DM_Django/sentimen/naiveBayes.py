from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from imblearn.over_sampling import SMOTE


class Naive:
    def model(df):

        # model
        model_g = BernoulliNB()
        vectorizer = TfidfVectorizer (max_features=3000)

        v_data = vectorizer.fit_transform(df['text']).toarray()
        X_train, X_test, y_train, y_test = train_test_split(v_data, df['sentimen'], test_size=0.03, random_state=0)
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        return model_g.fit(X_train, y_train)