echo "Start downloading embeddings.."
wget "https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-17.06.txt.gz"
echo "Embeddings downloaded. Start unzipping."
gunzip numberbatch-17.06.txt.gz
echo "Embeddings unzipped. Start filtering English and French words."
python extract_embeddings_conceptNet.py
echo "Done. Cleaning.."
rm numberbatch-17.06.txt
echo "Done!"
echo 
echo
echo 



