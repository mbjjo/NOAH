pip install virtualenv
cmd /k "cd /d ./lib & virtualenv NOAH_env & cd /d ./NOAH_env/Scripts & activate & pip install -r ../../requirements.txt & echo Installation complete"
cmd /ks