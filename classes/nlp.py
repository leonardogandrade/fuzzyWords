"""
       Module docstring
"""


import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np


class UniversalSentenceEncoder:
       def __init__(self) -> None:
              self.model = ''
              self.matches = []
       
       def loadModel(self):
              tf.keras.backend.clear_session()
              module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
              self.model = hub.load(module_url)
              
       def sentence_similarity(self, doc: list, query: list, min_corr: float = 0.64):
              bank_vec = self.model(doc)
              query_vec = self.model(query)
              correlation = np.transpose(np.inner(query_vec,bank_vec))
              
              correlation = np.transpose(np.inner(query_vec,bank_vec))
              print("Closest match found to '",query[0],"' is '", doc[np.argmax(correlation, axis=0)[0]],"'")
              print("Correlation matrix shape: ",correlation.shape)
              
              self.matches = []
              
              for corr_idx, corr in enumerate(correlation):
                     if corr >= min_corr:
                            self.matches.append(doc[corr_idx])
              
              # self.matches.append(doc[np.argmax(correlation, axis=0)[0]])
              result = dict()
              result['query'] = query
              result['match'] = self.matches
              result['corr_matrix'] = correlation.tolist()
              
              
              return result
       
   
# module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
# model = hub.load(module_url)
# print ("module %s loaded" % module_url)

# doc = ["I ate an hanburger",
#        "the hanburger is taste",
#        "hanburger is junkfood",
#        "My lunch was a big hanburger",
#        "do you like apple?"]

# query = ["I like to eat hamburger"]

# bank_vec = model(doc)

# query_vec = model(query)

# correlation = np.transpose(np.inner(query_vec,bank_vec))
# print("Closest match found to '",query[0],"' is '",doc[np.argmax(correlation, axis=0)[0]],"'")
# print("Correlation matrix shape: ",correlation.shape)