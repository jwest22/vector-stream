from models import SemanticClassifier

def parse_and_classify_query(query):
    text_query = []
    geo_query = []
    time_query = []
    token_classification = []

    tokens = query.split()
    classifier = SemanticClassifier()

    for token in tokens:
        classification = classifier.classify(token)
        if classification == 0:
            text_query.append(token)
            token_classification.append((token, 'description'))
        elif classification == 1:
            geo_query.append(token)
            token_classification.append((token, 'city'))
        elif classification == 2:
            time_query.append(token)
            token_classification.append((token, 'timestamp'))

    return " ".join(text_query), " ".join(geo_query), time_query, token_classification
