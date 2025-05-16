# Zadanie wstępne do Mastercard Fraud Detection Hackathon

### Aby zacząć utwórz środowisko witrtualne i pobierz requirements.txt
```bash
conda create -n hackathon python=3.10
conda activate hackathon
pip install -r requirements.txt
```

### Aby pobrać dane z kaggle trzeba utworzyć własny klucz API Kaggle

Wejdź na ustawienie profilu na stronie Kaggle
Kliknij "Utwórz nowy API Token"
Utwórz folder ".kaggle" w katalogu domowym i przenieś tam pobrany plik "kaggle.json"

```bash
mkdir ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

[Link do competition na Kaggle](https://www.kaggle.com/competitions/sgh-x-mastercard-hackathon-may-2025/data)

Trzeba do niego dołączyć, a następnie uruchomić skrypt download-data.py. 
Aby dane pobrały się we właściwym miejscu, naley uruchomić go wewnątrz katalogu `src/`