mkdir book_data/
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz
gunzip reviews_Books_5.json.gz
gunzip meta_Books.json.gz
python script/process_data.py meta_Books.json reviews_Books_5.json
python script/local_aggretor.py
python script/split_by_user.py
python script/generate_voc.py
mv local_train_splitByUser ./book_data/
mv local_test_splitByUser ./book_data/
