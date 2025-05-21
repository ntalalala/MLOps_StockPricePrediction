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

# Simple pickleable class for scaler tests
class SimpleScaler:
    def __init__(self, min_val, scale_val):
        self.min_ = np.array([min_val])
        self.scale_ = np.array([scale_val])

    def __eq__(self, other): # For easier comparison in tests if needed
        if not isinstance(other, SimpleScaler):
            return False
        return np.array_equal(self.min_, other.min_) and \
               np.array_equal(self.scale_, other.scale_)


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

        db_utils.setup_database(SAMPLE_DB_CONFIG) # Assumes db_utils.py is fixed

        executed_sql = [c[0][0].strip() for c in mock_cursor.execute.call_args_list]
        self.assertIn("CREATE TABLE IF NOT EXISTS raw_stock_data", executed_sql[0])
        self.assertIn("CREATE INDEX IF NOT EXISTS idx_raw_ticker_date ON raw_stock_data (ticker, date)", executed_sql[1])
        # ... (add asserts for other tables/indexes if needed, ensure order or use a set for unordered checks)
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS model_performance_log" in s for s in executed_sql))


        mock_conn.commit.assert_called_once()
        mock_cursor.close.assert_called_once() # Should pass if db_utils.py is fixed
        mock_conn.close.assert_called_once()   # Should pass if db_utils.py is fixed

    @patch('utils.db_utils.get_db_connection')
    def test_save_to_raw_table_success(self, mock_get_db_connection):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        data = {
            'Open': [np.float64(100.0), 101.0, pd.NA],
            'High': [102.0, 103.0, 104.0],
            'Low': [99.0, 100.0, 100.5],
            'Close': [101.5, 102.5, 103.5],
            'Volume': [np.int64(100000), 120000, np.int32(130000)],
            'Dividends': [0.0, np.float32(0.5), 0.0],
            'Stock Splits': [0.0, 0.0, 2.0]
        }
        dates = [pd.Timestamp('2023-01-01 00:00:00'), pd.Timestamp('2023-01-02 00:00:00'), pd.Timestamp('2023-01-03 00:00:00')]
        df = pd.DataFrame(data, index=pd.Index(dates, name='date'))
        
        ticker = 'TESTRAW'
        rows_affected = db_utils.save_to_raw_table(ticker, df, SAMPLE_DB_CONFIG)

        self.assertEqual(rows_affected, 3)
        mock_cursor.executemany.assert_called_once()
        args_list = mock_cursor.executemany.call_args[0][1]
        self.assertEqual(args_list[0][0], ticker)
        self.assertIsInstance(args_list[0][2], float)
        self.assertIsNone(args_list[2][2]) # pd.NA became None
        mock_conn.commit.assert_called_once()

    @patch('utils.db_utils.pd.read_sql_query')
    @patch('utils.db_utils.get_db_connection')
    def test_get_latest_raw_data_window(self, mock_get_db_connection, mock_read_sql):
        mock_conn = MagicMock()
        mock_get_db_connection.return_value = mock_conn
    
        raw_db_data_full = { # Data as if a larger set exists in DB, ordered DESC
            'date': [datetime(2023,1,3), datetime(2023,1,2), datetime(2023,1,1)],
            'open': [102.0,101.0,100.0], 'high': [103.0,102.0,101.0], 'low': [101.0,100.0,99.0],
            'close': [102.5,101.5,100.5], 'volume': [12000.0,11000.0,10000.0],
            'dividends': [0.0,0.0,0.0], 'stock_splits': [0.0,0.0,0.0]
        }
        full_mock_df_from_sql = pd.DataFrame(raw_db_data_full)
    
        tickers = ['WINRAW']
        window_size = 2 
    
        # Simulate the SQL LIMIT: mock_read_sql returns only 'window_size' rows
        # These would be the LATEST 'window_size' rows (top N from DESC sort)
        limited_mock_df_from_sql = full_mock_df_from_sql.head(window_size).copy()
        mock_read_sql.return_value = limited_mock_df_from_sql
    
        result_dict = db_utils.get_latest_raw_data_window(SAMPLE_DB_CONFIG, tickers, window_size)
    
        self.assertIn('WINRAW', result_dict)
        df_win = result_dict['WINRAW']
    
        # Expected df after db_utils processing: sorted ASC, correct columns, indexed by date
        # These are the two latest dates from raw_db_data_full, now sorted ASC
        expected_processed_data = {
            'Open': [101.0, 102.0], 'High': [102.0, 103.0], 'Low': [100.0, 101.0],
            'Close': [101.5, 102.5], 'Volume': [11000.0, 12000.0], # data for 2023-01-02 and 2023-01-03
            'Dividends': [0.0, 0.0], 'Stock Splits': [0.0, 0.0]
        }
        expected_dates = pd.to_datetime([datetime(2023,1,2), datetime(2023,1,3)])
        expected_df = pd.DataFrame(expected_processed_data, index=pd.Index(expected_dates, name='date'))
    
        mock_read_sql.assert_called_once_with(ANY, mock_conn, params=('WINRAW', window_size), parse_dates=['date'])
        pd.testing.assert_frame_equal(df_win, expected_df)


    @patch('utils.db_utils.get_db_connection')
    def test_save_and_load_scalers(self, mock_get_db_connection):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        run_id = "scaler_run_data_789"
        
        simple_scaler_x = SimpleScaler(0.1, 0.9)
        simple_y_scaler = SimpleScaler(100.0, 50.0)

        scalers_data = {
            'scalers_x': [[simple_scaler_x]],
            'y_scalers': [simple_y_scaler],
            'tickers': ['TICKA'],
            'num_features': 1
        }

        db_utils.save_scalers(SAMPLE_DB_CONFIG, run_id, scalers_data)
        args_tuple = mock_cursor.execute.call_args[0][1]
        self.assertEqual(args_tuple[0], run_id)
        
        loaded_pickle_content = pickle.loads(args_tuple[1])
        self.assertEqual(loaded_pickle_content.keys(), scalers_data.keys())
        self.assertEqual(loaded_pickle_content['tickers'], ['TICKA'])
        # Check equality of our SimpleScaler objects (thanks to __eq__)
        self.assertEqual(loaded_pickle_content['scalers_x'][0][0], simple_scaler_x)
        self.assertEqual(loaded_pickle_content['y_scalers'][0], simple_y_scaler)
        mock_conn.commit.assert_called_once()

        mock_cursor.fetchone.return_value = (pickle.dumps(scalers_data),)
        loaded_scalers_from_db = db_utils.load_scalers(SAMPLE_DB_CONFIG, run_id)
        
        self.assertEqual(loaded_scalers_from_db.keys(), scalers_data.keys())
        self.assertEqual(loaded_scalers_from_db['tickers'], ['TICKA'])
        self.assertEqual(loaded_scalers_from_db['scalers_x'][0][0], simple_scaler_x)
        self.assertEqual(loaded_scalers_from_db['y_scalers'][0], simple_y_scaler)

    # ... (Keep other tests as they were, assuming they passed or adapting them similarly if needed) ...
    # Example: test_check_ticker_exists, test_load_data_from_db, test_save_and_load_processed_features,
    # test_save_and_load_scaled_features, test_save_and_load_optimization_results, test_save_prediction,
    # test_get_latest_prediction_for_all_tickers, test_get_latest_target_date_prediction_for_ticker,
    # test_get_raw_stock_data_for_period, test_get_all_distinct_tickers_from_predictions,
    # test_save_daily_performance_metrics, test_get_recent_performance_metrics,
    # test_get_last_data_timestamp_for_ticker, test_get_prediction_for_date_ticker,
    # test_get_predictions_for_ticker_in_daterange

    # Minimal remaining tests for brevity, ensure you port all your tests over
    @patch('utils.db_utils.get_db_connection')
    def test_load_processed_features_not_found(self, mock_get_db_connection):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        loaded_data = db_utils.load_processed_features_from_db(SAMPLE_DB_CONFIG, "non_existent_run")
        self.assertIsNone(loaded_data)

    @patch('utils.db_utils.get_db_connection')
    def test_get_all_distinct_tickers_from_predictions(self, mock_get_db_connection):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [('AAPL',), ('MSFT',), ('GOOG',)]
        result = db_utils.get_all_distinct_tickers_from_predictions(SAMPLE_DB_CONFIG)
        expected = ['AAPL', 'MSFT', 'GOOG']
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
