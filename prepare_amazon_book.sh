mkdir book_data/
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz
gunzip reviews_Books_5.json.gz
gunzip meta_Books.json.gz
python process/preprocess.py meta_Books.json reviews_Books_5.json
python process/local_aggregator.py
python process/split.py
python process/voc_generator.py
mv local_train_splitByUser ./book_data/
mv local_test_splitByUser ./book_data/
