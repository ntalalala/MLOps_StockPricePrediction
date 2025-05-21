# tests/utils/test_db_utils.py

import unittest
from unittest.mock import patch, MagicMock, ANY, call
import psycopg2
import psycopg2.extras # For DictCursor
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, date, timedelta
from decimal import Decimal
from pathlib import Path
import sys

# Ensure src is in a discoverable path for tests
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import db_utils

# Sample DB config for tests
SAMPLE_DB_CONFIG = {
    'dbname': 'test_db',
    'user': 'test_user',
    'password': 'test_password',
    'host': 'localhost',
    'port': '5432'
}

class TestDBUtils(unittest.TestCase):

    @patch('utils.db_utils.psycopg2.connect')
    def test_get_db_connection_success(self, mock_connect):
        mock_conn_instance = MagicMock()
        mock_connect.return_value = mock_conn_instance
        conn = db_utils.get_db_connection(SAMPLE_DB_CONFIG)
        mock_connect.assert_called_once_with(
            dbname=SAMPLE_DB_CONFIG['dbname'],
            user=SAMPLE_DB_CONFIG['user'],
            password=SAMPLE_DB_CONFIG['password'],
            host=SAMPLE_DB_CONFIG['host'],
            port=SAMPLE_DB_CONFIG['port']
        )
        self.assertEqual(conn, mock_conn_instance)

    @patch('utils.db_utils.psycopg2.connect')
    def test_get_db_connection_failure(self, mock_connect):
        mock_connect.side_effect = psycopg2.OperationalError("Connection failed")
        with self.assertRaises(psycopg2.OperationalError):
            db_utils.get_db_connection(SAMPLE_DB_CONFIG)

    @patch('utils.db_utils.get_db_connection')
    def test_setup_database(self, mock_get_db_connection):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        db_utils.setup_database(SAMPLE_DB_CONFIG)

        # Verify all expected CREATE TABLE and CREATE INDEX statements
        executed_sql = [c[0][0].strip() for c in mock_cursor.execute.call_args_list] # Get the SQL part of each call

        self.assertIn("CREATE TABLE IF NOT EXISTS raw_stock_data", executed_sql[0]) # First call
        self.assertIn("CREATE INDEX IF NOT EXISTS idx_raw_ticker_date ON raw_stock_data (ticker, date)", executed_sql[1])
        self.assertIn("CREATE TABLE IF NOT EXISTS processed_feature_data", executed_sql[2])
        self.assertIn("CREATE TABLE IF NOT EXISTS scaled_feature_sets", executed_sql[3])
        self.assertIn("CREATE TABLE IF NOT EXISTS scalers", executed_sql[4])
        self.assertIn("CREATE TABLE IF NOT EXISTS latest_predictions", executed_sql[5])
        self.assertIn("CREATE INDEX IF NOT EXISTS idx_latest_predictions_ticker_target_date", executed_sql[6])
        self.assertIn("CREATE TABLE IF NOT EXISTS optimization_results", executed_sql[7])
        self.assertIn("CREATE TABLE IF NOT EXISTS model_performance_log", executed_sql[8])
        self.assertIn("CREATE INDEX IF NOT EXISTS idx_perf_log_date_ticker ON model_performance_log (prediction_date, ticker)", executed_sql[9])

        mock_conn.commit.assert_called_once()
        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch('utils.db_utils.get_db_connection')
    def test_save_to_raw_table_success(self, mock_get_db_connection):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Data closely matching yfinance output and db_utils expectations
        data = {
            'Open': [np.float64(100.0), 101.0, pd.NA], # Added pd.NA for NULL test
            'High': [102.0, 103.0, 104.0],
            'Low': [99.0, 100.0, 100.5],
            'Close': [101.5, 102.5, 103.5],
            'Volume': [np.int64(100000), 120000, np.int32(130000)],
            'Dividends': [0.0, np.float32(0.5), 0.0],
            'Stock Splits': [0.0, 0.0, 2.0] # Stock split can be int or float
        }
        dates = [pd.Timestamp('2023-01-01 00:00:00'), pd.Timestamp('2023-01-02 00:00:00'), pd.Timestamp('2023-01-03 00:00:00')]
        df = pd.DataFrame(data, index=pd.Index(dates, name='date'))
        
        ticker = 'TESTRAW'
        rows_affected = db_utils.save_to_raw_table(ticker, df, SAMPLE_DB_CONFIG)

        self.assertEqual(rows_affected, 3) # Number of rows processed
        mock_cursor.executemany.assert_called_once()
        
        # Check data types and values in the call to executemany
        args_list = mock_cursor.executemany.call_args[0][1] # List of tuples

        # Record 1
        self.assertEqual(args_list[0][0], ticker)                   # ticker
        self.assertEqual(args_list[0][1], dates[0])                 # date (Timestamp object)
        self.assertIsInstance(args_list[0][2], float)               # open
        self.assertEqual(args_list[0][2], 100.0)
        self.assertIsInstance(args_list[0][6], int)                 # volume
        self.assertEqual(args_list[0][6], 100000)
        self.assertIsInstance(args_list[0][7], float)               # dividends
        self.assertEqual(args_list[0][7], 0.0)
        self.assertIsInstance(args_list[0][8], float)               # stock_splits
        self.assertEqual(args_list[0][8], 0.0)

        # Record 2 (testing np.float32 conversion)
        self.assertIsInstance(args_list[1][7], float)               # dividends
        self.assertEqual(args_list[1][7], 0.5)

        # Record 3 (testing pd.NA to None and np.int32)
        self.assertIsNone(args_list[2][2])                          # open (was pd.NA)
        self.assertIsInstance(args_list[2][6], int)                 # volume
        self.assertEqual(args_list[2][6], 130000)

        mock_conn.commit.assert_called_once()

    @patch('utils.db_utils.get_db_connection')
    def test_check_ticker_exists(self, mock_get_db_connection):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (1,)
        self.assertTrue(db_utils.check_ticker_exists('AAPL', SAMPLE_DB_CONFIG))
        mock_cursor.fetchone.return_value = (0,)
        self.assertFalse(db_utils.check_ticker_exists('MSFT', SAMPLE_DB_CONFIG))

    @patch('utils.db_utils.pd.read_sql_query')
    @patch('utils.db_utils.get_db_connection')
    def test_load_data_from_db(self, mock_get_db_connection, mock_read_sql):
        mock_conn = MagicMock()
        mock_get_db_connection.return_value = mock_conn
        
        # Data as pd.read_sql_query would return it for raw_stock_data
        sample_db_return_data = {
            'date': [datetime(2023,1,1,0,0,0), datetime(2023,1,2,0,0,0)], # Note: time part might be 00:00:00
            'open': [100.0, 101.0], 'high': [102.0, 103.0], 'low': [99.0, 100.0],
            'close': [101.5, 102.5], 'volume': [100000.0, 120000.0], # Volume might be float from DB
            'dividends': [0.0, 0.0], 'stock_splits': [0.0, 0.0]
        }
        mock_df_from_sql = pd.DataFrame(sample_db_return_data)
        
        # Expected DataFrame after processing in db_utils
        expected_processed_df = mock_df_from_sql.copy()
        expected_processed_df['date'] = pd.to_datetime(expected_processed_df['date'])
        expected_processed_df = expected_processed_df.set_index('date')
        expected_processed_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']

        mock_read_sql.return_value = mock_df_from_sql.copy()

        tickers = ['AAPL']
        result = db_utils.load_data_from_db(SAMPLE_DB_CONFIG, tickers)

        mock_read_sql.assert_called_once_with(
            ANY, mock_conn, params=('AAPL',), parse_dates=['date']
        )
        self.assertIn('AAPL', result)
        pd.testing.assert_frame_equal(result['AAPL'], expected_processed_df)
        mock_conn.close.assert_called_once()

    @patch('utils.db_utils.get_db_connection')
    def test_save_and_load_processed_features(self, mock_get_db_connection):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        run_id = "test_run_proc_123"
        data_np = np.array([[[1.1, 2.2], [3.3, 4.4]]], dtype=np.float32) # Example with specific dtype
        targets_np = np.array([[5.5, 6.6]], dtype=np.float32)
        features = ['ema_50', 'rsi_14']
        tickers_list = ['STOCKA', 'STOCKB']

        db_utils.save_processed_features_to_db(SAMPLE_DB_CONFIG, data_np, targets_np, features, tickers_list, run_id)
        
        args_tuple = mock_cursor.execute.call_args[0][1] # (run_id, processed_data_blob, ...)
        self.assertEqual(args_tuple[0], run_id)
        # Compare after pickling and unpickling to ensure exact match of what's stored
        np.testing.assert_array_equal(pickle.loads(args_tuple[1]), data_np)
        np.testing.assert_array_equal(pickle.loads(args_tuple[2]), targets_np)
        self.assertEqual(json.loads(args_tuple[3]), features)
        self.assertEqual(json.loads(args_tuple[4]), tickers_list)
        mock_conn.commit.assert_called_once()

        mock_cursor.fetchone.return_value = (
            run_id, pickle.dumps(data_np), pickle.dumps(targets_np), 
            json.dumps(features), json.dumps(tickers_list)
        )
        loaded_data = db_utils.load_processed_features_from_db(SAMPLE_DB_CONFIG, run_id)
        
        self.assertIsNotNone(loaded_data)
        np.testing.assert_array_equal(loaded_data['processed_data'], data_np)
        np.testing.assert_array_equal(loaded_data['targets'], targets_np)
        self.assertEqual(loaded_data['feature_columns'], features)
        self.assertEqual(loaded_data['tickers'], tickers_list)

    @patch('utils.db_utils.get_db_connection')
    def test_save_and_load_scaled_features(self, mock_get_db_connection):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        run_id = "scaled_run_feat_456"
        set_name = "X_test_scaled"
        data_np = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float64)

        db_utils.save_scaled_features(SAMPLE_DB_CONFIG, run_id, set_name, data_np)
        args_tuple = mock_cursor.execute.call_args[0][1]
        self.assertEqual(args_tuple[0], run_id)
        self.assertEqual(args_tuple[1], set_name)
        np.testing.assert_array_equal(pickle.loads(args_tuple[2]), data_np)
        mock_conn.commit.assert_called_once()

        mock_cursor.fetchone.return_value = (pickle.dumps(data_np),)
        loaded_data = db_utils.load_scaled_features(SAMPLE_DB_CONFIG, run_id, set_name)
        np.testing.assert_array_equal(loaded_data, data_np)

    @patch('utils.db_utils.get_db_connection')
    def test_save_and_load_scalers(self, mock_get_db_connection):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        run_id = "scaler_run_data_789"
        # A more realistic scalers_dict structure (MinMaxScalers are not easily comparable with ==)
        # For unit test, we care about the structure being saved/loaded.
        # Actual scaler objects would be complex. We can mock their relevant attributes if needed for deeper tests.
        mock_scaler_x_stock0_feat0 = MagicMock() 
        mock_scaler_x_stock0_feat0.min_ = np.array([0.1])
        mock_scaler_x_stock0_feat0.scale_ = np.array([0.9])

        mock_y_scaler_stock0 = MagicMock()
        mock_y_scaler_stock0.min_ = np.array([100.0])
        mock_y_scaler_stock0.scale_ = np.array([50.0])


        scalers_data = {
            'scalers_x': [[mock_scaler_x_stock0_feat0]], # list of lists of scalers
            'y_scalers': [mock_y_scaler_stock0],       # list of scalers
            'tickers': ['TICKA'],
            'num_features': 1
        }

        db_utils.save_scalers(SAMPLE_DB_CONFIG, run_id, scalers_data)
        args_tuple = mock_cursor.execute.call_args[0][1]
        self.assertEqual(args_tuple[0], run_id)
        # For complex objects, we might just check type or key existence after loading
        # Deep comparison of pickled complex objects can be tricky.
        loaded_pickle_content = pickle.loads(args_tuple[1])
        self.assertEqual(loaded_pickle_content.keys(), scalers_data.keys())
        mock_conn.commit.assert_called_once()

        mock_cursor.fetchone.return_value = (pickle.dumps(scalers_data),)
        loaded_scalers = db_utils.load_scalers(SAMPLE_DB_CONFIG, run_id)
        self.assertEqual(loaded_scalers.keys(), scalers_data.keys())
        self.assertEqual(loaded_scalers['tickers'], ['TICKA'])
        # A more robust check for scaler objects would involve checking their attributes
        self.assertEqual(loaded_scalers['scalers_x'][0][0].min_, mock_scaler_x_stock0_feat0.min_)

    @patch('utils.db_utils.get_db_connection')
    def test_save_prediction(self, mock_get_db_connection):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        ticker = "PREDSAVE"
        price = 123.45 # float
        model_id = "model_v1_save"
        target_date_str = "2023-10-26" # YYYY-MM-DD string as per db_utils

        db_utils.save_prediction(SAMPLE_DB_CONFIG, ticker, price, model_id, target_date_str)
        
        # SQL is 'INSERT INTO latest_predictions (target_prediction_date, ticker, predicted_price, model_mlflow_run_id, prediction_logged_timestamp)...'
        # Parameters for VALUES: (target_prediction_date_str, ticker, predicted_price, model_mlflow_run_id)
        # The CURRENT_TIMESTAMP is handled by SQL.
        expected_params_tuple = (target_date_str, ticker, price, model_id)
        
        mock_cursor.execute.assert_called_once()
        actual_sql_call = mock_cursor.execute.call_args[0] # This is a tuple (sql_string, params_tuple)
        self.assertEqual(actual_sql_call[1], expected_params_tuple) # Compare the params tuple
        mock_conn.commit.assert_called_once()

    @patch('utils.db_utils.get_db_connection')
    def test_get_latest_prediction_for_all_tickers(self, mock_get_db_connection):
        mock_conn = MagicMock()
        # mock_cursor needs to behave like DictCursor
        mock_cursor = MagicMock(spec=psycopg2.extensions.cursor) 
        mock_get_db_connection.return_value = mock_conn
        # Ensure .cursor() method returns a factory that produces DictCursor-like mocks
        mock_conn.cursor.return_value = mock_cursor
        
        # Data as if returned by DictCursor from 'latest_predictions'
        mock_db_row1 = {'ticker': 'AAPL', 'target_prediction_date': date(2023,10,25), 'predicted_price': Decimal('150.75'), 'model_mlflow_run_id': 'run1'}
        mock_db_row2 = {'ticker': 'MSFT', 'target_prediction_date': date(2023,10,24), 'predicted_price': Decimal('300.50'), 'model_mlflow_run_id': 'run2'}
        # If fetchall is called, it returns a list of these dict-like rows
        mock_cursor.fetchall.return_value = [mock_db_row1, mock_db_row2]

        result = db_utils.get_latest_prediction_for_all_tickers(SAMPLE_DB_CONFIG)
        
        expected = [
            {'ticker': 'AAPL', 'date': '2023-10-25', 'predicted_price': 150.75, 'model_mlflow_run_id': 'run1'},
            {'ticker': 'MSFT', 'date': '2023-10-24', 'predicted_price': 300.50, 'model_mlflow_run_id': 'run2'}
        ]
        self.assertEqual(result, expected)
        mock_conn.cursor.assert_called_once_with(cursor_factory=psycopg2.extras.DictCursor)
        self.assertIn("SELECT DISTINCT ON (ticker)", mock_cursor.execute.call_args[0][0])

    @patch('utils.db_utils.get_db_connection')
    def test_get_latest_target_date_prediction_for_ticker(self, mock_get_db_connection):
        mock_conn = MagicMock()
        mock_cursor = MagicMock(spec=psycopg2.extensions.cursor)
        mock_get_db_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        ticker_sym = 'XYZTARG'
        # Case 1: Found, price is Decimal
        mock_db_row_decimal = {
            'target_prediction_date': date(2023,10,25), 
            'predicted_price': Decimal('180.20'), 
            'model_mlflow_run_id': 'run_xyz_dec'
        }
        mock_cursor.fetchone.return_value = mock_db_row_decimal
        result = db_utils.get_latest_target_date_prediction_for_ticker(SAMPLE_DB_CONFIG, ticker_sym)
        expected = {'target_prediction_date': '2023-10-25', 'predicted_price': 180.20, 'model_mlflow_run_id': 'run_xyz_dec'}
        self.assertEqual(result, expected)
        mock_conn.cursor.assert_called_with(cursor_factory=psycopg2.extras.DictCursor)
        mock_cursor.execute.assert_called_with(ANY, (ticker_sym.upper(),))


        # Case 2: Found, price is float
        mock_db_row_float = {
            'target_prediction_date': date(2023,10,26), 
            'predicted_price': 190.55, # Python float
            'model_mlflow_run_id': 'run_abc_flt'
        }
        mock_cursor.fetchone.return_value = mock_db_row_float
        result_float = db_utils.get_latest_target_date_prediction_for_ticker(SAMPLE_DB_CONFIG, 'ABC')
        expected_float = {'target_prediction_date': '2023-10-26', 'predicted_price': 190.55, 'model_mlflow_run_id': 'run_abc_flt'}
        self.assertEqual(result_float, expected_float)

        # Case 3: Found, price is None
        mock_db_row_none_price = {
            'target_prediction_date': date(2023,10,27), 
            'predicted_price': None, 
            'model_mlflow_run_id': 'run_def_none'
        }
        mock_cursor.fetchone.return_value = mock_db_row_none_price
        result_none_price = db_utils.get_latest_target_date_prediction_for_ticker(SAMPLE_DB_CONFIG, 'DEF')
        expected_none_price = {'target_prediction_date': '2023-10-27', 'predicted_price': None, 'model_mlflow_run_id': 'run_def_none'}
        self.assertEqual(result_none_price, expected_none_price)

        # Case 4: Not found
        mock_cursor.fetchone.return_value = None
        result_not_found = db_utils.get_latest_target_date_prediction_for_ticker(SAMPLE_DB_CONFIG, 'NOTFOUND')
        self.assertIsNone(result_not_found)

    @patch('utils.db_utils.pd.read_sql_query')
    @patch('utils.db_utils.get_db_connection')
    def test_get_raw_stock_data_for_period(self, mock_get_db_connection, mock_read_sql):
        mock_conn = MagicMock()
        mock_get_db_connection.return_value = mock_conn
        
        # Data as pd.read_sql_query would return it (date, close)
        sample_db_return_data = {
            'date': [datetime(2023,1,2,0,0,0), datetime(2023,1,1,0,0,0)], # Simulate DESC order from DB
            'close': [101.0, 100.0]
        }
        mock_df_from_sql = pd.DataFrame(sample_db_return_data)
        mock_read_sql.return_value = mock_df_from_sql.copy()

        end_dt_obj = date(2023,1,2)
        num_d = 2
        ticker_sym = 'RAWPER'
        result_df = db_utils.get_raw_stock_data_for_period(SAMPLE_DB_CONFIG, ticker_sym, end_dt_obj, num_d)

        mock_read_sql.assert_called_once_with(ANY, mock_conn, params=(ticker_sym.upper(), end_dt_obj, num_d))
        
        # Expected DataFrame after processing (sorted ascending by date)
        expected_df_data = {
            'date': pd.to_datetime([datetime(2023,1,1,0,0,0), datetime(2023,1,2,0,0,0)]),
            'close': [100.0, 101.0]
        }
        expected_df = pd.DataFrame(expected_df_data).reset_index(drop=True)
        
        # Ensure 'date' column is datetime for comparison
        result_df['date'] = pd.to_datetime(result_df['date'])
        
        pd.testing.assert_frame_equal(result_df.sort_values(by='date').reset_index(drop=True), expected_df)


    @patch('utils.db_utils.get_db_connection')
    def test_get_last_data_timestamp_for_ticker(self, mock_get_db_connection):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        dt_obj = datetime(2023, 10, 25, 16, 0, 0) # Example with time
        mock_cursor.fetchone.return_value = (dt_obj,)
        self.assertEqual(db_utils.get_last_data_timestamp_for_ticker(SAMPLE_DB_CONFIG, 'AAPLTS'), dt_obj)

        mock_cursor.fetchone.return_value = (None,) # DB returns (None,) if MAX is on no rows
        self.assertIsNone(db_utils.get_last_data_timestamp_for_ticker(SAMPLE_DB_CONFIG, 'NEWTTS'))
        
        mock_cursor.fetchone.return_value = None # fetchone itself returns None if query returns no rows
        self.assertIsNone(db_utils.get_last_data_timestamp_for_ticker(SAMPLE_DB_CONFIG, 'NEWTTS2'))

    @patch('utils.db_utils.pd.read_sql_query')
    @patch('utils.db_utils.get_db_connection')
    def test_get_latest_raw_data_window(self, mock_get_db_connection, mock_read_sql):
        mock_conn = MagicMock()
        mock_get_db_connection.return_value = mock_conn

        # Data as pd.read_sql_query would return it (cols from raw_stock_data, sorted DESC by date)
        raw_db_data = {
            'date': [datetime(2023,1,3), datetime(2023,1,2), datetime(2023,1,1)],
            'open': [102,101,100], 'high': [103,102,101], 'low': [101,100,99],
            'close': [102.5,101.5,100.5], 'volume': [12000.0,11000.0,10000.0],
            'dividends': [0,0,0], 'stock_splits': [0,0,0]
        }
        mock_df_from_sql = pd.DataFrame(raw_db_data)
        mock_read_sql.return_value = mock_df_from_sql.copy() # Query returns N latest rows

        tickers = ['WINRAW']
        window_size = 2 # We want the latest 2 for processing
        
        result_dict = db_utils.get_latest_raw_data_window(SAMPLE_DB_CONFIG, tickers, window_size)

        self.assertIn('WINRAW', result_dict)
        df_win = result_dict['WINRAW']
        
        # Expected df after processing: sorted ASC by date, correct columns, indexed by date
        expected_processed_data = {
            'Open': [101.0, 102.0], 'High': [102.0, 103.0], 'Low': [100.0, 101.0],
            'Close': [101.5, 102.5], 'Volume': [11000.0, 12000.0],
            'Dividends': [0.0, 0.0], 'Stock Splits': [0.0, 0.0]
        }
        expected_dates = pd.to_datetime([datetime(2023,1,2), datetime(2023,1,3)])
        expected_df = pd.DataFrame(expected_processed_data, index=pd.Index(expected_dates, name='date'))
        
        mock_read_sql.assert_called_once_with(ANY, mock_conn, params=('WINRAW', window_size), parse_dates=['date'])
        pd.testing.assert_frame_equal(df_win, expected_df)

    @patch('utils.db_utils.get_db_connection')
    def test_save_daily_performance_metrics(self, mock_get_db_connection):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        p_date_str = "2023-10-25" # String as per function signature
        ticker_sym = "PERFSAVE"
        metrics = {
            'actual_price': 100.0, 'predicted_price': 102.0, 'mae': 2.0, 
            'rmse': 2.0, 'mape': 0.02, 'direction_accuracy': 1.0
        }
        model_id = "perf_model_v1_save"

        db_utils.save_daily_performance_metrics(SAMPLE_DB_CONFIG, p_date_str, ticker_sym, metrics, model_id)
        
        # Order of params for 'INSERT INTO model_performance_log (prediction_date, ticker, actual_price, ...)'
        expected_call_params = (
            p_date_str, ticker_sym, 
            metrics['actual_price'], metrics['predicted_price'],
            metrics['mae'], metrics['rmse'], metrics['mape'], metrics['direction_accuracy'],
            model_id
        )
        mock_cursor.execute.assert_called_once()
        actual_sql_call_params = mock_cursor.execute.call_args[0][1] # The tuple of parameters
        self.assertEqual(actual_sql_call_params, expected_call_params)
        mock_conn.commit.assert_called_once()


if __name__ == '__main__':
    unittest.main()
