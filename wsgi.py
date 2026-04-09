from app import app

# WSGI entrypoint for production servers (gunicorn/waitress).
if __name__ == "__main__":
    app.run()
