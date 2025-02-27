import pandas as pd
from sklearn.metrics import average_precision_score
import numpy as np
import os

def get_ap(df, label_column, top_percentile=0.5):
    top = int(len(df)*top_percentile)
    df = df.sort_values('score', ascending=False).head(top)
    # after selecting top percentile candidates, we set the score for the dummy kp to 1, to prevent it from increasing the precision.
    df.loc[df['key_point_id'] == "dummy_id", 'score'] = 0.99
    ap = average_precision_score(y_true=df[label_column], y_score=df["score"])
    # multiply by the number of positives in top 50% and devide by the number of max positives within the top 50%, which is the number of top 50% instances
    positives_in_top_predictions = sum(df[label_column])
    max_num_of_positives = len(df)
    ap_retrieval = ap * positives_in_top_predictions/max_num_of_positives
    return ap_retrieval

def calc_mean_average_precision(df, label_column):
    precisions = [get_ap(group, label_column) for _, group in df.groupby(["topic", "stance"])]
    return np.mean(precisions)

def evaluate_predictions(merged_df):
    print("\n** running evalution:")
    mAP_strict = calc_mean_average_precision(merged_df, "label_strict")
    mAP_relaxed = calc_mean_average_precision(merged_df, "label_relaxed")
    print(f"mAP strict= {mAP_strict} ; mAP relaxed = {mAP_relaxed}")
    return mAP_strict, mAP_relaxed

def load_kpm_data(gold_data_dir, subset, submitted_kp_file=None):
    print("\nֿ** loading task data:")
    arguments_file = os.path.join(gold_data_dir, f"arguments_{subset}.csv")
    if not submitted_kp_file:
        key_points_file = os.path.join(gold_data_dir, f"key_points_{subset}.csv")
    else:
        key_points_file=submitted_kp_file
    labels_file = os.path.join(gold_data_dir, f"labels_{subset}.csv")


    arguments_df = pd.read_csv(arguments_file)
    key_points_df = pd.read_csv(key_points_file)
    labels_file_df = pd.read_csv(labels_file)

    return arguments_df, key_points_df, labels_file_df


def get_predictions(predictions_dict, labels_df, arg_df, kp_df):
    arg_df = arg_df[["arg_id", "topic", "stance"]]
    predictions_df = load_predictions(predictions_dict, kp_df["key_point_id"].unique())

    #make sure each arg_id has a prediction
    predictions_df = pd.merge(arg_df, predictions_df, how="left", on="arg_id")

    #handle arguements with no matching key point
    predictions_df["key_point_id"] = predictions_df["key_point_id"].fillna("dummy_id")
    predictions_df["score"] = predictions_df["score"].fillna(0)

    #merge each argument with the gold labels
    merged_df = pd.merge(predictions_df, labels_df, how="left", on=["arg_id", "key_point_id"])

    merged_df.loc[merged_df['key_point_id'] == "dummy_id", 'label'] = 0
    merged_df["label_strict"] = merged_df["label"].fillna(0)
    merged_df["label_relaxed"] = merged_df["label"].fillna(1)

    return merged_df


"""
this method chooses the best key point for each argument
and generates a dataframe with the matches and scores
"""
def load_predictions(predictions_dict, correct_kp_list):
    arg =[]
    kp = []
    scores = []
    invalid_keypoints = set()
        
    for arg_id, kps in predictions_dict.items():
        valid_kps = {key: value for key, value in kps.items() if key in correct_kp_list}
        invalid = {key: value for key, value in kps.items() if key not in correct_kp_list}
        # For all those kp that doesn't belong to that argument
        for invalid_kp, _ in invalid.items():
            if invalid_kp not in invalid_keypoints:
                print(f"key point {invalid_kp} doesn't appear in the key points file and will be ignored")
                invalid_keypoints.add(invalid_kp)

        if valid_kps:
            best_kp = max(valid_kps.items(), key=lambda x: x[1])
            arg.append(arg_id)
            kp.append(best_kp[0])
            scores.append(best_kp[1])
    print(f"\tloaded predictions for {len(arg)} arguments")
    return pd.DataFrame({"arg_id" : arg, "key_point_id": kp, "score": scores})
