wget http://www.openslr.org/resources/12/test-clean.tar.gz -P $PWD/data
wget http://www.openslr.org/resources/12/train-clean-100.tar.gz -P $PWD/data
tar -xvzf data/test-clean.tar.gz -C data
tar -xvzf data/train-clean-100.tar.gz -C data
rm data/test-clean.tar.gz
rm data/train-clean-100.tar.gz