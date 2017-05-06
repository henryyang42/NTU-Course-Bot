unzip ./crawler/review.zip -d crawler # ntu-course comment (from ptt)
unzip ./crawler/html.zip -d crawler # 2 years' course
pip3 install -r requirements.txt # install required packages

python3 manage.py makemigrations
python3 manage.py makemigrations crawler
python3 manage.py migrate
python3 manage.py createsuperuser

echo "Moving data to database..."
python3 misc_scripts/install_db.py
echo "Done."
python3 manage.py runserver
