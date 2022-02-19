import os
os.chdir(r'F:\LocalDriveD\Analytics\Learning\flask with dash\plotlydash-flask-tutorial')
from plotlyflask_tutorial import init_app

app = init_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0")