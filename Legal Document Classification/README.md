# Legal Document Classification  
This project classifies legal documents into categories using NLP.  

## Dataset  
- [Harvard Case Law Dataset](https://case.law/)
- [SEC Filings](https://www.sec.gov/edgar.shtml)  

## Approach  
- Preprocessing: Tokenization, stopword removal  
- Model: BERT-based classification  
- Evaluation: F1-score, AUC  

## Results  
Achieved **85% accuracy** using LegalBERT.  

## How to Run  
```bash
git clone https://github.com/your-username/ml-legal-portfolio.git  
cd legal-document-classification  
python train_model.py  
