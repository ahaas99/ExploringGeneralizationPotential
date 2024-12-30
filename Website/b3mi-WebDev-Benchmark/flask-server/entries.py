from app import app, db
from app.models import User, CSVFile, SubmissionMetrics, DatasetMetrics
from datetime import datetime, timezone
import pandas as pd
import os
import glob

USER_ID = 2
teamname = 'baseline'

def insert_submission_metrics():
    main_df = pd.read_csv(r'C:\Users\Marius\Desktop\Models-Plots\reevaluation-aggregated-metrics\average_over_datasets.csv')
    with app.app_context():
        for index, row in main_df.iterrows():
            model = row['model'] + '_endToEnd_baseline' # TODO + training regimen
            resolution = str(row['resolution']) + 'x' + str(row['resolution'])
            # Extract metrics
            accuracy_test = row['ACC_mean']
            auc_test = row['AUC_mean']
            balacc_test = row['BALACC_mean']
            co_test = row['Co_mean']
            prec_test = row['Prec_mean']

            # Create a new CSVFile entry
            csv_file = CSVFile(
                filename='baseline_upload.csv',
                teamname = teamname,
                user_id=USER_ID,
                upload_date=datetime.now(timezone.utc),
                study_link=None,
                model_name=model,
                description=f'Baseline entry for {model} and resolution {resolution}',
                resolution=resolution
            )
            db.session.add(csv_file)
            db.session.flush()

            # Create a new SubmissionMetrics entry
            submission_metrics = SubmissionMetrics(
                csv_file_id=csv_file.id,
                accuracy_test=accuracy_test,
                auc_test=auc_test,
                balacc_test=balacc_test,
                co_test=co_test,
                prec_test=prec_test
            )
            db.session.add(submission_metrics)
        db.session.commit()

def insert_dataset_metrics():
    dataset_csv_files = glob.glob(r'C:\Users\Marius\Desktop\Models-Plots\reevaluation-aggregated-metrics\*_aggregated_metrics.csv')
    with app.app_context():
        for dataset_csv_file in dataset_csv_files:
            dataset_name = dataset_name = os.path.basename(dataset_csv_file).split('_')[0]
            dataset_df = pd.read_csv(dataset_csv_file)
            
            for index, row in dataset_df.iterrows():
                training_regimen = row['trainingRegimen']
                model = row['model'] + '_endToEnd_baseline'
                for res in ['28', '64', '128', '224']:
                    acc_mean_col = f'ACC_{res}_mean'
                    auc_mean_col = f'AUC_{res}_mean'
                    balacc_mean_col = f'BALACC_{res}_mean'
                    co_mean_col = f'Co_{res}_mean'
                    prec_mean_col = f'Prec_{res}_mean'

                    if acc_mean_col in dataset_df.columns:
                        csv_file = CSVFile.query.filter_by(
                            model_name=model,
                            resolution=f'{res}x{res}'
                        ).order_by(CSVFile.upload_date.desc()).first()

                        if csv_file:
                            dataset_metric = DatasetMetrics(
                                csv_file_id=csv_file.id,
                                dataset_name=dataset_name,
                                test_acc = clean_value(row[acc_mean_col]),
                                test_AUC = clean_value(row[auc_mean_col]),
                                test_balacc = clean_value(row[balacc_mean_col]),
                                test_co = clean_value(row[co_mean_col]),
                                test_prec = clean_value(row[prec_mean_col])
                            )

                            try:
                                db.session.add(dataset_metric)
                            except Exception as e:
                                print(f"An error occurred: {e}")
                                db.session.rollback()  # Rollback the session if there's an error
                            
        db.session.commit()

def clean_value(value):
    return value if not pd.isna(value) else None

if __name__ == '__main__':
    insert_submission_metrics()
    insert_dataset_metrics()
