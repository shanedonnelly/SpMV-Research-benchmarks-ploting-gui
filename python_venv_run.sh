[ ! -d "venv" ] && python3 -m venv venv && ./venv/bin/pip install -r requirements.txt; cd ./app && ../venv/bin/streamlit run app.py && cd ..
