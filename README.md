# N-Gram-Model
A sentence generator to show at the BxSci Club Fair

### Usage instructions
Clone the repository
To run: python trigram_model.py data/brown_train.txt

### Explanation for how a random sentence is generated
The code scans the brown_train.txt corpus for sample sentences. It computes a distribution of 'trigrams' - pairs of three words, and their frequency within the text corpus. For example, there may be 20 instances of ('The', 'blue', 'cat') in the corpus. To generate a sentence, the first two tokens are ('Start', 'Start'). Then, the program looks for trigrams which it has seen starting with ('Start', 'Start') and adds the third word to the generated sentence (i.e. now the sentence may be: ('Start', 'Start', 'A'). Keep repeating, with the two tokens being shifted to the right: (i.e. ('Start', 'Start'), ('Start', 'Start, 'A'). ('Start', 'A') -> ('Start', 'A', 'nice'). ('A', 'nice') -> ('A', 'nice', 'day')). The loop continues until a 'Stop' token is added to the sentence.
