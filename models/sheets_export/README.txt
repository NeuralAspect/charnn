Google Sheets Character RNN Model

Files:
- Wxh.csv: Input to hidden weights
- Whh.csv: Hidden to hidden weights
- Why.csv: Hidden to output weights
- bh.csv: Hidden bias
- by.csv: Output bias
- vocab.csv: Character to index mapping

Instructions for Google Sheets:
1. Import each CSV file into separate sheets
2. For one-hot encoding, use: =IF(COLUMN()=B2+1,1,0) where B2 is the character index
3. For matrix multiplication, use: =MMULT()
4. For tanh, use: =TANH()
5. For softmax, use: =EXP(A1)/SUM(EXP(A1:A10)) where A1:A10 is your range
6. For random sampling, use: =RANDBETWEEN(1,100)/100

Note: You'll need to implement the full RNN logic using these basic operations.
