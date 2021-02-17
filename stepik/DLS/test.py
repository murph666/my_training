def process(sentences):
    result = []
    for elem in sentences:
        if elem.isalpha():
            result.append(elem)
        else:
            if not elem.isnumeric():
                result.append(' '.join([i for i in elem.split(' ') if i.isalpha()]))
    result.remove('')
    return result


def process2(sentences):
    result = []
    word = [x.split() for x in sentences]
    for el in word:
        el = list(filter(lambda x: x.isalpha(), el))
        result.append(' '.join(el))
    return result


# print(process(['1 thousand devils', 'My name is 9Pasha', 'Room #125 costs $100', '888', '888abc']))
print(process2(['1 thousand devils', 'My name is 9Pasha', 'Room #125 costs $100', '888', '888abc']))
