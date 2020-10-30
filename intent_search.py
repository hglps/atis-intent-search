import atisClassifier as classifier

# to search
# print(chatbot_query('when is the next flight'))

def chatbot_query(query, index=0):
    result = ''

    result = classifier.predictIntent(query)
    return result
