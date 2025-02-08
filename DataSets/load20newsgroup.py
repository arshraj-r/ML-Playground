import pandas as pd
from  sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

class NewsGroupLoader:
    def __init__(self , test_size=0.3, random_state=43):
        """initialise the class 
        """
        self.test_size=test_size
        self.random_state  =random_state
        self._load_data()

    def _load_data(self):
        """Dowloads all the 20newsgroup data and splits it into train and test"""
        news_group_data=fetch_20newsgroups(subset="all",remove=("headers","footers","quotes"))

        category_names=news_group_data.target_names

        X_train,X_test, y_train,y_test=train_test_split(
            news_group_data.data, news_group_data.target, test_size=self.test_size,random_state=self.random_state )
        self.train_df=pd.DataFrame({"text":X_train,"label":y_train})
        self.test_df=pd.DataFrame({"text":X_test,"label":y_test})

        self.train_df['category'] = self.train_df['label'].apply(lambda x: category_names[x])
        self.test_df['category'] = self.test_df['label'].apply(lambda x: category_names[x])

    
    def get_train_data(self):
        """returns the traing data as a dataframe """
        return self.train_df

    def get_test_data(self):
        """returns the test data as a dataframe"""
        return self.test_df

if __name__=="__main__":
    loader=NewsGroupLoader(test_size=.20)
    train_df=loader.get_train_data()
    # print(train_df.shape)
    print(train_df.head(3))

    test_df=loader.get_test_data()
    # print(test_df.shape)
    print(test_df.head(3))




