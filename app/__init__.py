from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with a secure key

    # Import and register blueprints
    from .routes import app_routes
    app.register_blueprint(app_routes)

    return app
