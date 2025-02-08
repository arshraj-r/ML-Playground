from sklearn.datasets import (
    load_iris, load_digits, load_diabetes, load_wine, load_breast_cancer,
    fetch_california_housing, fetch_openml, fetch_20newsgroups
)

class SklearnDatasetLoader:
    def __init__(self, dataset_name:str, as_dataframe=True):
        """I
        Initializes the dataset loader.

        :param datast_name: Name of the dataset.
        :param as_dataframe: Weather to return a pandas dataframe.
        """
        self.dataset_name=dataset_name
        self.as_dataframe=as_dataframe
    
    def load_data(self):
        """
        Maps the dataset based on the given name.
        """
        dataset_map = {
            "iris":load_iris,
            "digits": load_digits,
            "diabetes": load_diabetes,
            "wine": load_wine,
            "breast_cancer": load_breast_cancer,
            "california_housing": fetch_california_housing,
            "20newsgroups":fetch_20newsgroups
                    } 
        if self.dataset_name in dataset_map:
            data = dataset_map[self.dataset_name](as_frame=self.as_dataframe)
            return self._to_dataframe(data) if self.as_dataframe else (data.data, data.target)

        else:
            # Try loading from OpenML (useful for large datasets)
            return self._load_from_openml()

    def _load_from_openml(self):
        """Fetches dataset from OpenML if not in built-in datasets."""
        try:
            data = fetch_openml(name=self.dataset_name, as_frame=self.as_dataframe, parser="auto")
            return self._to_dataframe(data) if self.as_dataframe else (data.data, data.target)
        except Exception as e:
            print(f"Error fetching dataset from OpenML: {e}")
            return None

    def _to_dataframe(self, data):
        """Converts sklearn dataset to a Pandas DataFrame."""
        df = data.frame if hasattr(data, "frame") and data.frame is not None else pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df

# Example usage:
if __name__ == "__main__":
    loader = SklearnDatasetLoader("20newsgroups")
    df = loader.load_data()
    print(df.head())


