import pandas as pd

def check_submission(submission_path, test_path):
    """
    Performs basic sanity checks on a submission file.
    """
    print(f"Running sanity checks on {submission_path}...")
    try:
        sub = pd.read_csv(submission_path)
        test = pd.read_csv(test_path)
        
        # 1. Check for correct column names
        if not all(col in sub.columns for col in ['sample_id', 'price']):
            print("ERROR: Submission must have 'sample_id' and 'price' columns.")
            return False

        # 2. Check for correct number of rows
        if len(sub)!= len(test):
            print(f"ERROR: Submission has {len(sub)} rows, but test set has {len(test)} rows.")
            return False

        # 3. Check if all test sample_ids are present
        if not sub['sample_id'].equals(test['sample_id']):
            print("ERROR: sample_id column does not match the test set.")
            return False
            
        # 4. Check for negative prices
        if (sub['price'] < 0).any():
            print("ERROR: Submission contains negative price predictions.")
            return False
            
        # 5. Check for missing values
        if sub.isnull().values.any():
            print("ERROR: Submission contains missing values.")
            return False

        print("Sanity check PASSED!")
        return True
    except Exception as e:
        print(f"An error occurred during sanity check: {e}")
        return False

if __name__ == "__main__":
    # Run this script from the root directory of your project
    check_submission('submissions/final_ensemble_submission.csv', 'data/test.csv')