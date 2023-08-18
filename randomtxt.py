import random

dictionary_file = '/usr/share/dict/linux.words'

with open(dictionary_file) as f:
    words = f.read().splitlines() 

num_sentences = 10
sentence_length = 8

for i in range(num_sentences):
    sentence = []
    for j in range(sentence_length):
        word = random.choice(words)
        sentence.append(word)
    
    print(' '.join(sentence))
