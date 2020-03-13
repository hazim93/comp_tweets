import gzip
import gensim 
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# with open('training.1600000.processed.noemoticon.csv', 'r') as f:
#     for i,line in enumerate (f):
#         print(line.split('"')[11])
#             break


def read_input(input_file):    
    logging.info("reading file {0}...this may take a while".format(input_file))
    
    with open (input_file, 'r') as f:
        for i, line in enumerate (f): 

            if (i%10000==0):
                logging.info ("read {0} reviews".format (i))
            # do some pre-processing and return a list of words for each review text
            yield gensim.utils.simple_preprocess (line.split('"')[11])


data_file = 'training.1600000.processed.noemoticon.csv'
# read the tokenized reviews into a list
# each review item becomes a serries of words
# so this becomes a list of lists
documents = list (read_input (data_file))
logging.info ("Done reading data file")


model = gensim.models.Word2Vec (documents, size=150, window=10, min_count=2, workers=10)
model.train(documents,total_examples=len(documents),epochs=10)


# w1 = "promo"
w1 = ['lunch','hour']
model.wv.most_similar (positive=w1)

