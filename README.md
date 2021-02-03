# Task realization outline (among others)
1. Transfer from clusterization problem to classification
2. Build nn model and validate it on the dev set
3. Achieved approx. 16 ari, hyperparameters and architecture can be enhanced, so the model can do a better job

# Data input pipeline
1. Normalize to proper unicode and lowercase
2. Razdel tokenization
3. Maru normalization (linear tagger and pymorphy lemmatizer) without assigning a pos tag

# Data to model (for more information see loader.py)
1. Fed in three ways (simultaneously):
  a. Full encoded context (for input to the rnn block)
  b. Embedded current word (which is being disambiguated) 
  c. Normalized summ of surrounding words with fixed window size (=10)
  d. Real length are provided as well
2. Embedded with lemmatized ELMo embeddings of dimension 1024 (which is then reduced to 300)
  
# Model overview (for more information see model.py)
1. Dropout on the embedded sequences (full encoded context)
2. Rnn-block (3-layers bidirectional lstm)
3. Dot-product soft global attention
4. Concatenation of last hidden states with b and c parts 
5. Dropout + dense
6. Weight decay is used on the last dense layer to prevent overfitting on averge CE

# How to run inference
1. You can do this running run() function from script.py file inside the project directory (in this case you use preprocessed stored data)
2. You can preprocess data from scratch and feed this for inference in the notebook solution.py in the last section again inside project directory
3. Do not forget install dependencies from requirements.txt file with simple pip install -r requirements.txt
4. Pretrained elmo model can donwloaded inside working directory from the notebook with wget command and appropriate link

# Obvious drawbacks:
1. Model is overfitted on the dev set (if it is supposed to be used in other processes as ready project)
2. Score can be higher at least by increasing model complexity (of course being aware of overfitting)

# Subtle advantages:
1. TF graph is optimized for inference (no gradient calculation nodes and operation + lower precision calculation)
2. Can be even more optimized (frozen graph)
3. Separate graphs and session are used for training and evaluation
