# Docker: Build if need , Run interactively, Stop (Ctrl+C), and auto-remove container
[ -z "$(docker images -q spmv-gui:latest)" ] && docker build --no-cache -t spmv-gui:latest . ; docker run -it --rm -p 8501:8501 -v $(pwd)/app/csv:/app/csv -v $(pwd)/app/pickle:/app/pickle -v $(pwd)/app/subset_pickle:/app/subset_pickle spmv-gui
# and if after you need to remove the image:
docker rmi spmv-gui:latest
# Docker: Build, Run interactively, Stop (Ctrl+C), and auto-remove container AND image
[ -z "$(docker images -q spmv-gui:latest)" ] && docker build --no-cache -t spmv-gui . && (trap 'docker rmi spmv-gui' EXIT; docker run -it --rm -p 8501:8501 -v $(pwd)/app/csv:/app/csv -v $(pwd)/app/pickle:/app/pickle -v $(pwd)/app/subset_pickle:/app/subset_pickle spmv-gui)

# Python Venv: Create venv if needed, install deps, Run, Stop (Ctrl+C)
[ ! -d "venv" ] && python3 -m venv venv && ./venv/bin/pip install -r requirements.txt; cd ./app && ../venv/bin/streamlit run app.py && cd ..

# Python Venv: Create venv, install deps, Run, Stop (Ctrl+C), and auto-remove venv
(trap 'rm -rf venv' EXIT;[ ! -d "venv" ] &&  python3 -m venv venv; ./venv/bin/pip install -r requirements.txt; cd ./app && ../venv/bin/streamlit run app.py && cd ..)
