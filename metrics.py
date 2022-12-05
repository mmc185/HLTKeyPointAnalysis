from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score

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