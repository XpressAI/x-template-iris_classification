if [[ "$OSTYPE" =~ ^msys ]]; then 
    OS=Scripts
else
    OS=bin
fi &&
python -m venv venv &&
source venv/$OS/activate &&
python -m pip install --upgrade pip &&
pip install -r requirements.txt
