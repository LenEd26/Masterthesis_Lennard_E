import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from features.CorrExtractor import CorrExtractor

class FeatureLoader():
    """loads features according to specified labels, and list of features,
       encodes categorical labels / features
    """
    
    def __init__(self, features_dir, corr_threshold):
        self.ce = CorrExtractor(corr_threshold)
        all_features_df = pd.read_csv(features_dir)

        # rename columns to avoid trouble
        all_features_df.rename(columns={
            'height(cm)':'height_cm',
            'weight(kg)':'weight_kg',
            'leg_length(cm)':'leg_length_cm'
        }, inplace=True)

        # encode categorical values as integers, 'inverse_transform' can be used to get the info back
        self.enc_sub = OrdinalEncoder()
        sub_array = all_features_df.loc[:,'sub'].to_numpy().reshape(-1, 1)
        all_features_df['sub'] = self.enc_sub.fit_transform(sub_array)

        self.enc_sex = OrdinalEncoder()
        sex_array = all_features_df.loc[:,'sex'].to_numpy().reshape(-1, 1)
        all_features_df['sex'] = self.enc_sex.fit_transform(sex_array)

        self.enc_cond = OrdinalEncoder()
        cond_array = all_features_df.loc[:,"condition"].to_numpy().reshape(-1, 1)
        all_features_df["condition"] = self.enc_cond.fit_transform(cond_array)

        self.enc_fatigue = OrdinalEncoder()
        fatigue_array = all_features_df.loc[:,"fatigue"].to_numpy().reshape(-1, 1)
        all_features_df["fatigue"] = self.enc_fatigue.fit_transform(fatigue_array)

        self.all_features_df = all_features_df

    def get_gait_features(self, features_list, target_label, corr_filter):
        """load all gait parameter features and target label(s), filter out highly correlated features

        Args:
            features_list (list of str): list of features to be selected
            target_label (list of str): list of target labels, can be one or multiple
            corr_filter (Boolean): whether to remove highly correlated features

        Returns:
            all_df (DataFrame): dataframe of selected features and labels
            gait_features_df (DataFrame): dataframe of selected features
            label_df (DataFrame): dataframe of selected labels
        """

        labels_list = ["sub"]
        labels_list.extend(target_label)
        all_columns = features_list + labels_list
        all_df = self.all_features_df[all_columns].copy()  # filter selected features and labels

        all_df = all_df.dropna(axis=0)  # drop NaNs
        all_df.reset_index(inplace=True, drop=True)

        label_df = all_df[labels_list].copy()

        if corr_filter:
            # remove highly correlated features
            gait_features_df = self.ce.transform(
                all_df[features_list].copy()
            )  
            all_df = pd.concat([gait_features_df, label_df], axis=1)
        else:
            gait_features_df = all_df[features_list].copy()

        return all_df, gait_features_df, label_df

    # def filter_labels(self, target_label):
    #     """get only label of interest, the other label is baseline

    #     Args:
    #         target_label (str): label of interest, i.e., "condition" or "fatigue". Only baseline 
    #         values of the other label will be kept (i.e., control or single-task)
    #     """
    #     if target_label == "fatigue":
    #         # get only single task data
    #         self.all_features_df = self.all_features_df[(self.all_features_df["condition"] == 1)] 
    #         self.all_features_df = self.all_features_df.drop(columns=["condition"])
    #     elif target_label == "condition":
    #         # get only control data
    #         self.all_features_df = self.all_features_df[(self.all_features_df["fatigue"] == 0)] 
    #         self.all_features_df = self.all_features_df.drop(columns=["fatigue"])
