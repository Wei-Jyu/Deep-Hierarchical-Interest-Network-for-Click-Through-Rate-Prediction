mkdir book_data/
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz
gunzip reviews_Books_5.json.gz
gunzip meta_Books.json.gz
python data_process/preprocess.py meta_Books.json reviews_Books_5.json
python data_process/local_aggregator.py
python data_process/split.py
python data_process/voc_generator.py
mv local_train_splitByUser ./book_data/
mv local_test_splitByUser ./book_data/
