from flask import Flask, request, jsonify, make_response, send_file, session
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from sqlalchemy.exc import IntegrityError
from os import environ
import pandas as pd
from markupsafe import escape
from dotenv import load_dotenv
from app.evaluate_results import computeMetrics, computeMetrics_corrupted
from werkzeug.utils import secure_filename
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import numpy as np
import zipfile
from flask_migrate import Migrate, upgrade, migrate, init
# from datetime import timedelta

load_dotenv()

from app import app, db  # import app and db from the init file
from app.models import User, CSVFile, DataRow, SubmissionMetrics, DatasetMetrics, DatasetMetricsCorrupted

print('ONE'*5)

bcrypt = Bcrypt(app)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = environ.get('SECRET_KEY')
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # 'Lax' or 'None' if using HTTPS
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True if using HTTPS
app.config['SESSION_COOKIE_DOMAIN'] = None

CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}}, supports_credentials=True)

print('TWO'*5)

@app.errorhandler(500)
def handle_500(error):
    response = jsonify({"status": "error", "message": str(error)})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response, 500
CORRUPTIONS_DS_KEYS = {
                'pathmnist': [
                    'pixelate', 'jpeg_compression', 'defocus_blur', 'motion_blur',
                    'brightness_up', 'brightness_down', 'contrast_up', 'contrast_down',
                    'saturate', 'stain_deposit', 'bubble'
                ],
                'bloodmnist': [
                    'pixelate', 'jpeg_compression', 'defocus_blur', 'motion_blur',
                    'brightness_up', 'brightness_down', 'contrast_up', 'contrast_down',
                    'saturate', 'stain_deposit', 'bubble'
                ],
                'dermamnist': [
                    'pixelate', 'jpeg_compression', 'gaussian_noise', 'speckle_noise',
                    'impulse_noise', 'shot_noise', 'defocus_blur', 'motion_blur',
                    'zoom_blur', 'brightness_up', 'brightness_down', 'contrast_up',
                    'contrast_down', 'black_corner', 'characters'
                ],
                'retinamnist': [
                    'pixelate', 'jpeg_compression', 'gaussian_noise', 'speckle_noise',
                    'defocus_blur', 'motion_blur', 'brightness_down', 'contrast_down'
                ],
                'tissuemnist': [
                    'pixelate', 'jpeg_compression', 'impulse_noise', 'gaussian_blur',
                    'brightness_up', 'brightness_down', 'contrast_up', 'contrast_down'
                ],
                'octmnist': [
                    'pixelate', 'jpeg_compression', 'speckle_noise', 'defocus_blur',
                    'motion_blur', 'contrast_down'
                ],
                'breastmnist': [
                    'pixelate', 'jpeg_compression', 'speckle_noise', 'motion_blur',
                    'brightness_up', 'brightness_down', 'contrast_down'
                ],
                'chestmnist': [
                    'pixelate', 'jpeg_compression', 'gaussian_noise', 'speckle_noise',
                    'impulse_noise', 'shot_noise', 'gaussian_blur', 'brightness_up',
                    'brightness_down', 'contrast_up', 'contrast_down', 'gamma_corr_up',
                    'gamma_corr_down'
                ],
                'pneumoniamnist': [
                    'pixelate', 'jpeg_compression', 'gaussian_noise', 'speckle_noise',
                    'impulse_noise', 'shot_noise', 'gaussian_blur', 'brightness_up',
                    'brightness_down', 'contrast_up', 'contrast_down', 'gamma_corr_up',
                    'gamma_corr_down'
                ],
                'organamnist': [
                    'pixelate', 'jpeg_compression', 'gaussian_noise', 'speckle_noise',
                    'impulse_noise', 'shot_noise', 'gaussian_blur', 'brightness_up',
                    'brightness_down', 'contrast_up', 'contrast_down', 'gamma_corr_up',
                    'gamma_corr_down'
                ],
                'organcmnist': [
                    'pixelate', 'jpeg_compression', 'gaussian_noise', 'speckle_noise',
                    'impulse_noise', 'shot_noise', 'gaussian_blur', 'brightness_up',
                    'brightness_down', 'contrast_up', 'contrast_down', 'gamma_corr_up',
                    'gamma_corr_down'
                ],
                'organsmnist': [
                    'pixelate', 'jpeg_compression', 'gaussian_noise', 'speckle_noise',
                    'impulse_noise', 'shot_noise', 'gaussian_blur', 'brightness_up',
                    'brightness_down', 'contrast_up', 'contrast_down', 'gamma_corr_up',
                    'gamma_corr_down'
                ]
            }
# registration route
@app.route('/register', methods=['POST'])
def register_user():
    email = request.json['email']
    teamname = request.json['teamname']
    password = bcrypt.generate_password_hash(request.json['password']).decode('utf-8')
    new_user = User(teamname=teamname, email=email, password=password)
    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"status": "success"})
    except IntegrityError:
        db.session.rollback()
        return jsonify({"status": "failed", "message": "User already exists"}), 400

# login route
@app.route('/login', methods=['POST'])
def login_user():
    try:
        user_login = request.json['login']  # 'login' can be either teamname or email
        password = request.json['password']
        user = User.query.filter((User.teamname == user_login) | (User.email == user_login)).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['loggedin'] = True
            session['id'] = user.id
            session['email'] = user.email
            session['teamname'] = user.teamname
            print(session)

            response_data = {
                'status': 'success',
                'message': 'Login successful',
                'id': escape(user.id),
                'teamname': escape(user.teamname)
            }

            return make_response(jsonify(response_data), 200)
        else:
            return jsonify({"status": "failed"}), 401
    
    except Exception as e:
        app.logger.error(f'Error during login: {e}')
        return jsonify({"status": "error", "message": str(e)}), 500

# profile route
@app.route('/profile', methods=['GET'])
def get_profile():
    print('THREE'*5)
    if 'loggedin' in session:
        user = User.query.get(session['id'])
        if user:
            response_data = {
                'id': user.id,
                'email': user.email,
                'teamname': user.teamname,
            }
            return jsonify(response_data), 200
        else:
            return jsonify({"status": "failed", "message": "User not found"}), 404
    else:
        return jsonify({"status": "failed", "message": "Not logged in"}), 401

# file upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'loggedin' not in session:
        return jsonify({'status': 'failed', 'message': 'Not logged in'}), 401
    
    if 'file' not in request.files:
        return jsonify({'status': 'failed', 'message': 'No file part'}), 400

    file = request.files['file']

    # If the user does not select a file
    if file.filename == '':
        return jsonify({'status': 'failed', 'message': 'No selected file'}), 400

    study_link = request.form.get('studyLink')
    model_name = request.form.get('modelName')
    description = request.form.get('description')
    resolution = request.form.get('resolution')
    dataset = request.form.get('dataset')
    if dataset == "MedMNIST-C":
    # Save the file to the server
        if file and file.filename.endswith('.csv'):
            try:
                # Secure the filename
                filename = secure_filename(file.filename)

                df = pd.read_csv(file)  # 'file' is a file-like object
                # You can now perform your computations on 'df'

                # Save CSVFile information to the database
                csv_file = CSVFile(
                    filename=filename,
                    user_id=session['id'],
                    teamname = session['teamname'],
                    upload_date=datetime.now(timezone.utc),
                    study_link=study_link,
                    model_name=model_name,
                    description=description,
                    resolution=resolution,
                    dataset=dataset
                )
                db.session.add(csv_file)
                db.session.flush()  # To get csv_file.id
                df_existing = pd.DataFrame(columns=['average_accuracy', 'average_auc', 'average_balanced_accuracy', 'average_cohen_kappa', 'average_precision'])

                # Save DataRows to the database
                #data_rows = []
                #for _, row in df.iterrows():
                #    data_row = DataRow(
                #        dataset_name=row['dataset_name'],
                #        prediction=row['prediction'],
                #        truth=row['truth'],
                #        probability=row['probability'],
                #        csv_file_id=csv_file.id
                #    )
                #    data_rows.append(data_row)
                #db.session.bulk_save_objects(data_rows)
                results = computeMetrics_corrupted(df)
                # Save SubmissionMetrics
               

                dataset_metrics_objects =[]
                for dataset in [ 'breastmnist','bloodmnist', 'chestmnist', 'dermamnist', 'octmnist', 'organamnist', 'organcmnist',
                    'organsmnist', 'pathmnist', 'pneumoniamnist', 'retinamnist', 'tissuemnist']:
             
                    
                    corruptions = CORRUPTIONS_DS_KEYS[dataset]
                    dataset_metrics_objects_corrupted = []
                   
                    for corruption in corruptions:
                        if (resolution == "28x28" and (
                        corruption == "stain_deposit" or corruption == "bubble" or corruption == "characters")):
                            pass
                        elif (resolution == "64x64" and (
                                corruption == "characters")):
                            pass
                        elif ((resolution != "224x224" and (
                                corruption == "motion_blur"))):
                            pass
                        else:
                        
                            for severity in range(5):
                            
                            
                                # Save DatasetMetrics
                                dataset_name = dataset + corruption + str(severity)
                                dataset_metric_corrupted = DatasetMetricsCorrupted(
                                        csv_file_id=csv_file.id,
                                        dataset_name=dataset,
                                        corruption=corruption,
                                        severity=severity,
                                        test_acc=results[dataset_name + '_test_acc'],
                                        test_AUC=results[dataset_name + '_test_AUC'],
                                        test_balacc=results[dataset_name + '_test_balacc'],
                                        test_co=results[dataset_name + '_test_co'],
                                        test_prec= 0 #results[dataset_name + '_test_prec']
                                    )
                                dataset_metrics_objects_corrupted.append(dataset_metric_corrupted)
                                db.session.bulk_save_objects(dataset_metrics_objects_corrupted)

                    acc_csv = [value for key, value in results.items() if dataset in key and '_test_acc' in key]
                    auc_csv = [value for key, value in results.items() if dataset in key and '_test_AUC' in key]
                    balanced_acc_csv = [value for key, value in results.items() if dataset in key and '_test_balacc' in key]
                    co_csv = [value for key, value in results.items() if dataset in key and '_test_co' in key]
                    prec_csv = [value for key, value in results.items() if dataset in key and '_test_prec' in key]


                    average_acc = np.mean(acc_csv)
                    average_auc = np.mean(auc_csv)
                    average_balanced_acc = np.mean(balanced_acc_csv)
                    average_co = np.mean(co_csv)
                    average_prec = np.mean(prec_csv)

                    dataset_metric = DatasetMetrics(
                        csv_file_id=csv_file.id,
                        dataset_name=dataset,
                        test_acc=average_acc,
                        test_AUC=average_auc,
                        test_balacc=average_balanced_acc,
                        test_co=average_co,
                        test_prec= 0 #average_prec
                    )
                    dataset_metrics_objects.append(dataset_metric)
                    db.session.bulk_save_objects(dataset_metrics_objects)
                    average_results = {
                        'average_accuracy': average_acc,
                        'average_auc': average_auc,
                        'average_balanced_accuracy': average_balanced_acc,
                        'average_cohen_kappa': average_co,
                        'average_precision': average_prec
                    }

                    # Convert the dictionary to a DataFrame
                    df_averages = pd.DataFrame([average_results])

                    # Concatenate the new DataFrame with the existing one
                    df_existing = pd.concat([df_existing, df_averages], ignore_index=True)


                submission_metrics = SubmissionMetrics(
                    csv_file_id=csv_file.id,
                    accuracy_test=df_existing["average_accuracy"].mean(),
                    auc_test=df_existing["average_auc"].mean(),
                    balacc_test=df_existing["average_balanced_accuracy"].mean(),
                    co_test=df_existing["average_cohen_kappa"].mean(),
                    prec_test= 0 # df_existing["average_precision"].mean()
                )
                db.session.add(submission_metrics)
                dataset_metrics_objects = []
                # Commit the transaction
                db.session.commit()

                print(results)
                return jsonify({
                    'status': 'success',
                    'message': 'File received and processed',
                }), 200
            except Exception as e:
                db.session.rollback()
                app.logger.error(f'Error during file upload: {e}')
                return jsonify({'status': 'failed', 'message': str(e)}), 500

        else:
            return jsonify({'status': 'failed', 'message': 'Invalid file format'}), 400
    else: 
        if file and file.filename.endswith('.csv'):
            try:
                # Secure the filename
                filename = secure_filename(file.filename)

                df = pd.read_csv(file)  # 'file' is a file-like object
                # You can now perform your computations on 'df'

                # Save CSVFile information to the database
                csv_file = CSVFile(
                    filename=filename,
                    user_id=session['id'],
                    teamname = session['teamname'],
                    upload_date=datetime.now(timezone.utc),
                    study_link=study_link,
                    model_name=model_name,
                    description=description,
                    resolution=resolution,
                    dataset=dataset
                )
                db.session.add(csv_file)
                db.session.flush()  # To get csv_file.id

                # Save DataRows to the database
                data_rows = []
                for _, row in df.iterrows():
                    data_row = DataRow(
                        dataset_name=row['dataset_name'],
                        prediction=row['prediction'],
                        truth=row['truth'],
                        probability=row['probability'],
                        csv_file_id=csv_file.id
                    )
                    data_rows.append(data_row)
                db.session.bulk_save_objects(data_rows)

                results = computeMetrics(df)

                # Save SubmissionMetrics
                submission_metrics = SubmissionMetrics(
                    csv_file_id=csv_file.id,
                    accuracy_test=results['accuracy_test'],
                    auc_test=results['auc_test'],
                    balacc_test=results['balacc_test'],
                    co_test=results['co_test'],
                    prec_test= 0 # results['prec_test']
                )
                db.session.add(submission_metrics)

                # Save DatasetMetrics
                dataset_metrics_objects = []
                for dataset_name in results['dataset_names']:
                    dataset_metric = DatasetMetrics(
                        csv_file_id=csv_file.id,
                        dataset_name=dataset_name,
                        test_acc=results[dataset_name + '_test_acc'],
                        test_AUC=results[dataset_name + '_test_AUC'],
                        test_balacc=results[dataset_name + '_test_balacc'],
                        test_co=results[dataset_name + '_test_co'],
                        test_prec= 0 #results[dataset_name + '_test_prec']
                    )
                    dataset_metrics_objects.append(dataset_metric)
                db.session.bulk_save_objects(dataset_metrics_objects)

                # Commit the transaction
                db.session.commit()

                print(results)
                return jsonify({
                    'status': 'success',
                    'message': 'File received and processed',
                }), 200
            except Exception as e:
                db.session.rollback()
                app.logger.error(f'Error during file upload: {e}')
                return jsonify({'status': 'failed', 'message': str(e)}), 500

        else:
            return jsonify({'status': 'failed', 'message': 'Invalid file format'}), 400
        #Implement to hande the data for corruptuions

# logout route
@app.route('/logout')
def logout_user():
    try:
         # Remove session data, this will log the user out
        session.clear()
        # Redirect to login page
        return jsonify({"status": "success", "message": "Logged out successfully"}), 200
    
    except Exception as e:
        app.logger.error(f'Error during logout: {e}')
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/user_submissions', methods=['GET'])
def get_user_submissions():
    if 'loggedin' in session:
        user_id = session['id']

        # Query the user's CSVFiles and related SubmissionMetrics
        submissions = (
            db.session.query(CSVFile, SubmissionMetrics)
            .join(SubmissionMetrics, CSVFile.id == SubmissionMetrics.csv_file_id)
            .filter(CSVFile.user_id == user_id)
            .all()
        )

        # Serialize the data
        submissions_data = []
        for csv_file, metrics in submissions:
            submissions_data.append({
                'csv_file_id': csv_file.id,
                'upload_date': csv_file.upload_date.isoformat(),
                'resolution': csv_file.resolution,
                'model_name': csv_file.model_name,
                'test_auc': metrics.auc_test,
                'test_balacc': metrics.balacc_test,
                'test_co': metrics.co_test,
                'dataset': csv_file.dataset,
            })

        return jsonify({'status': 'success', 'submissions': submissions_data})
    else:
        return jsonify({'status': 'failed', 'message': 'Not logged in'}), 401

@app.route('/generate_plots', methods=['POST'])
def generate_plots():
    if 'loggedin' not in session:
        return jsonify({'status': 'failed', 'message': 'Not logged in'}), 401

    data = request.get_json()
    submission_id1 = data.get('submissionId1')
    submission_id2 = data.get('submissionId2')  # Could be None

    if not submission_id1:
        return jsonify({'status': 'failed', 'message': 'First submission ID is required'}), 400

    try:
        # Retrieve dataset-wise metrics for submission_id1
        csv_file1 = CSVFile.query.get(submission_id1)
        if not csv_file1:
            return jsonify({'status': 'failed', 'message': f'Submission ID {submission_id1} not found'}), 404

        # Fetch DatasetMetrics for submission_id1
        metrics1 = DatasetMetrics.query.filter_by(csv_file_id=submission_id1).all()
        if not metrics1:
            return jsonify({'status': 'failed', 'message': f'No metrics found for Submission ID {submission_id1}'}), 404

        df1 = pd.DataFrame([{
            'dataset_name': m.dataset_name,
            'test_AUC': m.test_AUC,
            'test_balacc': m.test_balacc,
            'test_co': m.test_co
        } for m in metrics1])

        # Initialize variables for submission 2
        df2 = None
        model_name2 = None
        if submission_id2:
            # Retrieve dataset-wise metrics for submission_id2
            csv_file2 = CSVFile.query.get(submission_id2)
            if not csv_file2:
                return jsonify({'status': 'failed', 'message': f'Submission ID {submission_id2} not found'}), 404

            metrics2 = DatasetMetrics.query.filter_by(csv_file_id=submission_id2).all()
            if not metrics2:
                return jsonify({'status': 'failed', 'message': f'No metrics found for Submission ID {submission_id2}'}), 404

            df2 = pd.DataFrame([{
                'dataset_name': m.dataset_name,
                'test_AUC': m.test_AUC,
                'test_balacc': m.test_balacc,
                'test_co': m.test_co
            } for m in metrics2])
            model_name2 = csv_file2.model_name

        # Generate plots using the per-dataset metrics
        plots = generate_plots_from_metrics(df1, df2, csv_file1.model_name, model_name2)

        # Create an in-memory zip file
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for plot_name, plot_data in plots.items():
                zf.writestr(plot_name, plot_data.getvalue())

        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='plots.zip'
        )

    except Exception as e:
        app.logger.error(f'Error during plot generation: {e}', exc_info=True)
        return jsonify({'status': 'failed', 'message': str(e)}), 500
    
@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    try:
        # Get query parameters
        resolution = request.args.get('resolution')  # Now resolution is always provided
        metric = request.args.get('metric', 'auc_test')  # Default to 'auc_test' if not provided
        dataset = request.args.get('dataset')
        # Validate metric parameter
        valid_metrics = ['auc_test', 'balacc_test', 'co_test']
        if metric not in valid_metrics:
            return jsonify({'status': 'failed', 'message': 'Invalid metric parameter'}), 400

        # Ensure resolution parameter is provided
        if not resolution:
            return jsonify({'status': 'failed', 'message': 'Resolution parameter is required'}), 400

        # Build the query
        query = db.session.query(
            CSVFile.id.label('id'),
            User.teamname.label('teamname'),
            CSVFile.study_link.label('study_link'),
            CSVFile.model_name.label('model_name'),
            CSVFile.description.label('description'),
            getattr(SubmissionMetrics, metric).label('metric_value')
        ).join(SubmissionMetrics, SubmissionMetrics.csv_file_id == CSVFile.id
        ).join(User, User.id == CSVFile.user_id
        ).filter(getattr(SubmissionMetrics, metric) != None)

        # Apply resolution filter (now mandatory)
        query = query.filter(CSVFile.resolution == resolution)

        query = query.filter(CSVFile.dataset == dataset)

        # Order by the specified metric descending
        query = query.order_by(getattr(SubmissionMetrics, metric).desc())

        # Limit to 10 results
        results = query.limit(10).all()

        # Prepare response data
        data = []
        for row in results:
            data.append({
                'id': row.id,
                'teamname': row.teamname,
                'study_link': row.study_link if row.study_link else '-',
                'model_name': row.model_name,
                'description': row.description,
                'metric_value': row.metric_value
            })

        # If no results, return a special message
        if not data:
            return jsonify({'status': 'success', 'message': 'No matching submissions', 'data': []}), 200

        return jsonify({'status': 'success', 'data': data}), 200

    except Exception as e:
        app.logger.error(f'Error during leaderboard retrieval: {e}', exc_info=True)
        return jsonify({'status': 'failed', 'message': str(e)}), 500


def generate_plots_from_metrics(df1, df2=None, model_name1='Model 1', model_name2='Model 2'):
    import io

    # Ensure datasets are in the same order for both DataFrames
    if df2 is not None:
        # Merge dataframes to ensure alignment
        df_combined = pd.merge(df1, df2, on='dataset_name', how='inner', suffixes=('_1', '_2'))
        if df_combined.empty:
            raise ValueError('No matching datasets found between the two submissions.')
    else:
        df_combined = df1.copy()
        # Rename metric columns to have '_1' suffix
        df_combined.rename(columns={
            'test_AUC': 'test_AUC_1',
            'test_balacc': 'test_balacc_1',
            'test_co': 'test_co_1'
        }, inplace=True)
        # Add columns for the second model with None values
        df_combined['test_AUC_2'] = None
        df_combined['test_balacc_2'] = None
        df_combined['test_co_2'] = None

    # Metrics to plot
    metrics = ['test_AUC', 'test_balacc', 'test_co']

    # Initialize dictionary to store plots
    plot_data = {}

    for metric in metrics:
        # Line Plot
        plt.figure(figsize=(12, 6))

        # Prepare data for plotting line chart
        data_line = {
            'Dataset': df_combined['dataset_name'].tolist(),
            'Metric Value': df_combined[f'{metric}_1'].tolist(),
            'Model': [model_name1] * len(df_combined)
        }

        if df2 is not None:
            data_line['Dataset'] += df_combined['dataset_name'].tolist()
            data_line['Metric Value'] += df_combined[f'{metric}_2'].tolist()
            data_line['Model'] += [model_name2] * len(df_combined)

        plot_df_line = pd.DataFrame(data_line)

        # Handle missing values
        plot_df_line.dropna(subset=['Metric Value'], inplace=True)

        # Define custom seaborn palette
        if df2 is not None:
            palette = {
                model_name1: 'blue',
                model_name2: 'orange'
            }
        else:
            palette = {
                model_name1: 'blue'
            }

        # Plot Line Chart with Seaborn
        sns.lineplot(
            data=plot_df_line,
            x='Dataset',
            y='Metric Value',
            hue='Model',
            marker='o',
            palette=palette
        )
        plt.xticks(rotation=90)
        plt.title(f'Comparison of {metric.replace("test_", "").upper()} across Datasets')
        plt.tight_layout()

        # Save Line Plot to BytesIO
        img_data_line = io.BytesIO()
        plt.savefig(img_data_line, format='png')
        img_data_line.seek(0)
        plot_data[f'{metric.replace("test_", "")}_lineplot.png'] = img_data_line
        plt.close()

        # Box Plot
        plt.figure(figsize=(8, 6))

        # Prepare data for boxplot
        data_box = []

        # Data for first submission
        df_box_1 = pd.DataFrame({
            'Metric Value': df_combined[f'{metric}_1'],
            'Model': model_name1
        })
        data_box.append(df_box_1)

        if df2 is not None:
            # Data for second submission
            df_box_2 = pd.DataFrame({
                'Metric Value': df_combined[f'{metric}_2'],
                'Model': model_name2
            })
            data_box.append(df_box_2)

        # Combine data for boxplot
        plot_df_box = pd.concat(data_box, ignore_index=True)

        # Handle missing values
        plot_df_box.dropna(subset=['Metric Value'], inplace=True)

        # Plot Boxplot with Seaborn
        sns.boxplot(
            data=plot_df_box,
            x='Model',
            y='Metric Value',
            palette=palette
        )
        plt.title(f'Boxplot of {metric.replace("test_", "").upper()}')
        plt.tight_layout()

        # Save Boxplot to BytesIO
        img_data_box = io.BytesIO()
        plt.savefig(img_data_box, format='png')
        img_data_box.seek(0)
        plot_data[f'{metric.replace("test_", "")}_boxplot.png'] = img_data_box
        plt.close()

    return plot_data





if __name__ == "__main__":
    with app.app_context():
        print('ZERO'*5)
        # Generate migration (you can pass any message to the migration command)
        db.create_all()
       # ensure all tables are created
    app.run(debug=True, host='localhost')