from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate
from datetime import datetime, timezone

app = Flask(__name__)

# Set the configuration for the SQLAlchemy URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:xAIMedicalBenchmark24!@localhost/med-database'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = True

db = SQLAlchemy(app)
migrate = Migrate(app, db)

CORS(app, resources={r"/*": {"origins": "*"}})
