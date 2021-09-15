from pathlib import Path
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier


## Models
# define, list models to train on

def get_feature_table(datafile_path=None):
    """ Loads datafile as feature table or generates new one """
    if Path(datafile_path).is_file():
        # load feature table
        pass
    else:
        # import function form features.py to create new table
        pass


def train_model(model, feature_table):
    """ Train model on features from feature_table 
    
    Args:
        model: classification model
        feature_table (pd.DataFrame): Feature table with rows representing
            articles and containing a credibility feature to predict.
    Returns:
        trained model
    """
    pass


def evaluate_model(model):
    """ Generates stats, feature importance, confusion matrix on model """
    pass