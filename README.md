# quick_cdiscount
Quick 50% accuracy deep learning model for Cdiscount datascience competition at https://www.datascience.net/fr/challenge/20/details

This shortly describes how to reach ~50% accuracy on the Cdiscount challenge using deep learning techniques only. The steps are as follows:

1- Learn a 200-D word2vec model for the objects descriptions

I use https://github.com/beniz/word2vec but the output formats are compatible with that of the original Google implementation.

The binary file for the full word2vec model is available [cdis.bin.bz2](http://www.deepdetect.com/stuff/cdis/cdis.bin.bz2) [3Gb]

2- Add up the vectors for Description, Libelle and Marque features, this yields a 200-D description that fully replaces all text in initial dataset

The script preproc_w2v.py does this. Beware, it uses Gensim's word2vec reader, and it may require patching due to a UTF8 mismatch. To fix this, replace Gensim's word2vec.py line 914 with
```
word = b''.join(word)
```

Based on this script, you can use the word vectors as you want for your own purposes and algorithms in the competition.

3- Balancing the training examples so that every category is represented somewhere between the mean and the median

The resulting training file is shuffled, and 70K training examples have been removed to be used as a validation set, available here: [training file](http://www.deepdetect.com/stuff/cdis/train_w2v_balanced.csv.bz2) [5.8Gb/16Gb uncompressed], [validation file](http://www.deepdetect.com/stuff/cdis/validate_w2v.csv.bz2), [test file](http://www.deepdetect.com/stuff/cdis/test_w2v.csv.bz2)

4- Train a 1500x750x750 neural net with PReLu activations and dropout (0.5) for ~1M iterations
  
  This yields a ~34.7% F1-score with 77.5% accuracy, which yields ~50% accuracy on the leaderboard

For this purpose I use [deepdetect](https://github.com/beniz/deepdetect), an Open Source server and API built on top of [caffe](https://github.com/BVLC/caffe). You would need a GPU for this to run fast enough.

After starting deepdetect, training the neural network takes two lines:
```
curl -X PUT "http://localhost:8080/services/cdis" -d "{\"mllib\":\"caffe\",\"description\":\"cdis service\",\"type\":\"supervised\",\"parameters\":{\"input\":{\"connector\":\"csv\"},\"mllib\":{\"template\":\"mlp_db\",\"nclasses\":5794,\"layers\":[1500,750,750],\"activation\":\"prelu\"}},\"model\":{\"templates\":\"../templates/caffe/\",\"repository\":\"/path/to/your/model/repo\"}}"

curl -X POST "http://localhost:8080/train" -d "{\"service\":\"cdis\",\"async\":true,\"parameters\":{\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":1000000,\"test_interval\":10000,\"snapshot\":50000,\"base_lr\":0.01,\"test_initialization\":false},\"net\":{\"batch_size\":500}},\"input\":{\"db\":true,\"label\":\"Categorie3\",\"id\":\"Identifiant_Produit\",\"separator\":\";\",\"scale\":true},\"output\":{\"measure\":[\"acc\",\"mcll\",\"f1\"]}},\"data\":[\"train_w2v_balanced.csv\",\"validate_w2v.csv\"]}"
```

It should be fairly simple to reproduce with another neural network package.

There are many ways to improve on the results above, from modifying the network topology, to balancing the data and using the word vectors in different manners.

5- predict on test set by using the predict.py script
