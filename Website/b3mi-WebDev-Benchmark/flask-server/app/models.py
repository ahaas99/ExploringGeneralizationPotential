from . import db
from datetime import datetime, timezone


class User(db.Model):
    __tablename__="user"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)  # Primary Key
    email = db.Column(db.String(345), unique=True, nullable=False)
    teamname = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password = db.Column(db.String(60), nullable=False)
    create_time = db.Column(db.DateTime(timezone=True), default=datetime.now(timezone.utc))

    # Relationships
    csv_files = db.relationship('CSVFile', backref='user', lazy=True, foreign_keys='CSVFile.user_id', cascade="all, delete-orphan",
        passive_deletes=True)

    # for debugging, defines string representation of the User object
    def __repr__(self):
            return f'<User {self.teamname}>'

# define the structure of a row of the results .csv-file which is uploaded
class DataRow(db.Model):
    __tablename__ = 'data_rows'
    id = db.Column(db.Integer, primary_key=True)
    dataset_name = db.Column(db.String(50))
    prediction = db.Column(db.Text)
    truth = db.Column(db.Text)
    probability = db.Column(db.Text)
    csv_file_id = db.Column(db.Integer, db.ForeignKey('csv_files.id', ondelete='CASCADE'), nullable=False, index=True)

class CSVFile(db.Model):
    __tablename__ = 'csv_files'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    teamname = db.Column(db.String(255), db.ForeignKey('user.teamname'), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    upload_date = db.Column(db.DateTime(timezone=True), default=datetime.now(timezone.utc))
    study_link = db.Column(db.String(500))
    model_name = db.Column(db.String(255))
    description = db.Column(db.Text)
    resolution = db.Column(db.String(10))
    dataset = db.Column(db.String(20), default="MedMNIST+")

    # Relationships
    data_rows = db.relationship('DataRow', backref='csv_file', lazy=True, cascade="all, delete-orphan", passive_deletes=True)
    submission_metrics = db.relationship('SubmissionMetrics', uselist=False, backref='csv_file', cascade="all, delete-orphan", passive_deletes=True)
    dataset_metrics = db.relationship('DatasetMetrics', backref='csv_file', lazy=True, cascade="all, delete-orphan", passive_deletes=True)
    user_teamname = db.relationship('User', foreign_keys=[teamname], backref='csv_files_by_teamname')





class SubmissionMetrics(db.Model):
    __tablename__ = 'submission_metrics'
    id = db.Column(db.Integer, primary_key=True)
    csv_file_id = db.Column(db.Integer, db.ForeignKey('csv_files.id', ondelete='CASCADE'), nullable=False, unique=True)
    accuracy_test = db.Column(db.Float)
    auc_test = db.Column(db.Float)
    balacc_test = db.Column(db.Float)
    co_test = db.Column(db.Float)
    prec_test = db.Column(db.Float)


class DatasetMetrics(db.Model):
    __tablename__ = 'dataset_metrics'
    id = db.Column(db.Integer, primary_key=True)
    csv_file_id = db.Column(db.Integer, db.ForeignKey('csv_files.id', ondelete='CASCADE'), nullable=False)
    dataset_name = db.Column(db.String(50))
    test_acc = db.Column(db.Float)
    test_AUC = db.Column(db.Float)
    test_balacc = db.Column(db.Float)
    test_co = db.Column(db.Float)
    test_prec = db.Column(db.Float)

class DatasetMetricsCorrupted(db.Model):
    __tablename__ = 'dataset_metrics_corrupted'
    id = db.Column(db.Integer, primary_key=True)
    csv_file_id = db.Column(db.Integer, db.ForeignKey('csv_files.id', ondelete='CASCADE'), nullable=False)
    dataset_name = db.Column(db.String(50))
    corruption = db.Column(db.String(50))
    severity = db.Column(db.Integer)
    test_acc = db.Column(db.Float)
    test_AUC = db.Column(db.Float)
    test_balacc = db.Column(db.Float)
    test_co = db.Column(db.Float)
    test_prec = db.Column(db.Float)