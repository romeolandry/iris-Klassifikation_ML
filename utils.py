def match_predicion (prediction, match_class):
    list_prediction = []
    for val in prediction:
        list_prediction.append(match_class.get(val))
    return list_prediction