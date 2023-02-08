mkdir data data/data_isbi_2015/
cd data/data_isbi_2015
wget --no-check-certificate https://smart-stats-tools.org/sites/default/files/lesion_challenge/training_final_v4.zip -O a.zip
wget --no-check-certificate https://smart-stats-tools.org/sites/default/files/lesion_challenge/testdata_website_2016-03-24.zip -O b.zip
unzip a.zip
unzip b.zip
rm -rf a.zip b.zip
