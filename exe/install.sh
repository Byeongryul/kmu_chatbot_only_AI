if test -d myvenv
then
    echo "exist"
    . myvenv/Scripts/activate
else
    echo "not exist"
    python -m venv myvenv
    . myvenv/Scripts/activate
fi

pip install --upgrade -r requirements.txt