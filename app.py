from pathlib import Path
import hashlib
import json
import threading
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / 'sales_data.csv'
MODEL_FILE = BASE_DIR / 'model_bundle.joblib'
PROCESSED_FILE = BASE_DIR / 'sales_data_processed.csv'
EVAL_FILE = BASE_DIR / 'model_evaluation.csv'
FORECAST_FILE = BASE_DIR / 'inventory_recommendation.csv'
MODEL_INFO_FILE = BASE_DIR / 'model_info.json'
DATA_INFO_FILE = BASE_DIR / 'data_info.json'
UPLOAD_DIR = BASE_DIR / 'du_lieu_cap_nhat'
UPLOAD_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
SERVICE_Z = 1.65
AUTO_CHECK_SECONDS = 300
TRAINING_LOCK = threading.Lock()

FEATURES = [
    'Store ID', 'Product ID', 'Category', 'Seasonality',
    'Month', 'DayOfWeek', 'WeekOfYear', 'DayOfYear', 'DayOfMonth', 'Quarter',
    'IsWeekend', 'IsMonthStart', 'IsMonthEnd',
    'SinYear', 'CosYear', 'SinWeek', 'CosWeek', 'SinMonth',
    'Demand_Lag_1', 'Demand_Lag_2', 'Demand_Lag_3',
    'Demand_Lag_7', 'Demand_Lag_14', 'Demand_Lag_21', 'Demand_Lag_28',
    'Demand_Lag_30', 'Demand_Lag_56', 'Demand_Lag_90', 'Demand_Lag_180', 'Demand_Lag_365',
    'Demand_Mean_3', 'Demand_Mean_7', 'Demand_Mean_14', 'Demand_Mean_28',
    'Demand_Mean_56', 'Demand_Mean_90', 'Demand_Mean_365',
    'Demand_Median_7', 'Demand_Median_28',
    'Demand_Std_7', 'Demand_Std_28', 'Demand_Std_56', 'Demand_Std_90',
    'Demand_EWM_7', 'Demand_EWM_28', 'Demand_EWM_90',
    'Demand_Trend_7_28', 'Demand_Trend_28_90',
    'Sold_Lag_1', 'Sold_Lag_7', 'Sold_Mean_7',
    'Price', 'Discount'
]
CATEGORICAL = ['Store ID', 'Product ID', 'Category', 'Seasonality']
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]


def season_from_month(month):
    if month in [12, 1, 2]: return 'Đông'
    if month in [3, 4, 5]: return 'Xuân'
    if month in [6, 7, 8]: return 'Hè'
    return 'Thu'


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_data(path=DATA_FILE):
    if not Path(path).exists():
        raise FileNotFoundError(f'Không tìm thấy {Path(path).name}.')
    df = pd.read_csv(path)
    required = ['Date', 'Store ID', 'Product ID', 'Category', 'Inventory Level',
                'Units Sold', 'Units Ordered', 'Price', 'Discount', 'Seasonality', 'Demand']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError('Thiếu cột bắt buộc: ' + ', '.join(missing))
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).drop_duplicates()
    numeric_cols = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Price', 'Discount', 'Demand']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(df[c].median())
    for c in ['Store ID', 'Product ID', 'Category', 'Seasonality']:
        df[c] = df[c].fillna(df[c].mode().iloc[0]).astype(str)
    df = df.sort_values(['Store ID', 'Product ID', 'Date']).reset_index(drop=True)
    return df


def make_features(df, drop_na=True):
    df = df.copy()
    g = ['Store ID', 'Product ID']
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['DayOfMonth'] = df['Date'].dt.day
    df['Quarter'] = df['Date'].dt.quarter
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    df['SinYear'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
    df['CosYear'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)
    df['SinWeek'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['CosWeek'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['SinMonth'] = np.sin(2 * np.pi * df['Month'] / 12)
    for lag in [1, 2, 3, 7, 14, 21, 28, 30, 56, 90, 180, 365]:
        df[f'Demand_Lag_{lag}'] = df.groupby(g)['Demand'].shift(lag)
    shifted = df.groupby(g)['Demand'].shift(1)
    for w in [3, 7, 14, 28, 56, 90, 365]:
        df[f'Demand_Mean_{w}'] = shifted.groupby([df['Store ID'], df['Product ID']]).transform(lambda x: x.rolling(w, min_periods=w).mean())
    for w in [7, 28]:
        df[f'Demand_Median_{w}'] = shifted.groupby([df['Store ID'], df['Product ID']]).transform(lambda x: x.rolling(w, min_periods=w).median())
    for w in [7, 28, 56, 90]:
        df[f'Demand_Std_{w}'] = shifted.groupby([df['Store ID'], df['Product ID']]).transform(lambda x: x.rolling(w, min_periods=w).std())
    for span in [7, 28, 90]:
        df[f'Demand_EWM_{span}'] = df.groupby(g)['Demand'].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    df['Demand_Trend_7_28'] = df['Demand_Mean_7'] - df['Demand_Mean_28']
    df['Demand_Trend_28_90'] = df['Demand_Mean_28'] - df['Demand_Mean_90']
    df['Sold_Lag_1'] = df.groupby(g)['Units Sold'].shift(1)
    df['Sold_Lag_7'] = df.groupby(g)['Units Sold'].shift(7)
    sold_shifted = df.groupby(g)['Units Sold'].shift(1)
    df['Sold_Mean_7'] = sold_shifted.groupby([df['Store ID'], df['Product ID']]).transform(lambda x: x.rolling(7, min_periods=7).mean())
    df = df.replace([np.inf, -np.inf], np.nan)
    if drop_na:
        df = df.dropna(subset=FEATURES + ['Demand']).reset_index(drop=True)
    return df


def prepare_lgbm(frame, categories=None):
    x = frame[FEATURES].copy()
    for c in CATEGORICAL:
        x[c] = x[c].astype('category')
        if categories is not None and c in categories:
            x[c] = x[c].cat.set_categories(categories[c])
    return x


def make_sklearn_pipeline(model):
    prep = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
        ('num', 'passthrough', NUMERIC)
    ])
    return Pipeline([('prep', prep), ('model', model)])


def metrics(y_true, pred, name):
    y_true = np.asarray(y_true)
    pred = np.maximum(0, np.asarray(pred))
    nonzero = y_true != 0
    return {
        'Mô hình': name,
        'MAE': round(float(mean_absolute_error(y_true, pred)), 4),
        'RMSE': round(float(np.sqrt(mean_squared_error(y_true, pred))), 4),
        'MAPE': round(float(np.mean(np.abs((y_true[nonzero] - pred[nonzero]) / y_true[nonzero])) * 100), 4) if nonzero.any() else 0,
        'R2': round(float(r2_score(y_true, pred)), 4)
    }


def benchmark_models(train, test):
    """So sánh các mô hình đủ mạnh nhưng vẫn phù hợp để tự động huấn luyện định kỳ."""
    X_train_lgb = prepare_lgbm(train)
    X_test_lgb = prepare_lgbm(test)
    y_train = train['Demand']
    y_test = test['Demand']
    models = []

    # 1. LightGBM: thường rất phù hợp với dữ liệu bảng + chuỗi thời gian.
    lgbm = LGBMRegressor(
        objective='regression', n_estimators=2200, learning_rate=0.018,
        num_leaves=63, max_depth=-1, min_child_samples=20,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=2.0,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
    )
    lgbm.fit(X_train_lgb, y_train, categorical_feature=CATEGORICAL)
    models.append(('LightGBM', lgbm, metrics(y_test, lgbm.predict(X_test_lgb), 'LightGBM'), 'lgbm'))

    # 2. XGBoost: mô hình boosting thứ hai để kiểm chứng chéo kết quả.
    xgb = make_sklearn_pipeline(XGBRegressor(
        n_estimators=1000, max_depth=7, learning_rate=0.035,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=3.0, objective='reg:squarederror',
        tree_method='hist', random_state=RANDOM_STATE, n_jobs=-1
    ))
    xgb.fit(train[FEATURES], y_train)
    models.append(('XGBoost', xgb, metrics(y_test, xgb.predict(test[FEATURES]), 'XGBoost'), 'xgb'))

    # 3. HistGradientBoosting: nhẹ, nhanh, làm mô hình đối chứng.
    hist = make_sklearn_pipeline(HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=2.0, random_state=RANDOM_STATE
    ))
    hist.fit(train[FEATURES], y_train)
    models.append(('Gradient Boosting', hist, metrics(y_test, hist.predict(test[FEATURES]), 'Gradient Boosting'), 'hist'))


    return models

def train_model(force=False):
    with TRAINING_LOCK:
        if MODEL_FILE.exists() and not force:
            try:
                return joblib.load(MODEL_FILE)
            except Exception:
                pass
        raw = load_data(DATA_FILE)
        data = make_features(raw)
        dates = sorted(data['Date'].unique())
        split_date = dates[int(len(dates) * 0.80)]
        train = data[data['Date'] < split_date].copy()
        test = data[data['Date'] >= split_date].copy()
        candidates = benchmark_models(train, test)
        ranked = sorted([x for x in candidates if x[3] != 'baseline'], key=lambda x: (x[2]['MAE'], x[2]['RMSE']))
        best_name, _, best_eval, best_type = ranked[0]

        # Huấn luyện lại mô hình thắng trên toàn bộ dữ liệu để dự báo tương lai.
        if best_type == 'lgbm':
            best_model = LGBMRegressor(
                objective='regression', n_estimators=2200, learning_rate=0.018,
                num_leaves=63, max_depth=-1, min_child_samples=20,
                subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=2.0,
                random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
            )
            X_all = prepare_lgbm(data)
            best_model.fit(X_all, data['Demand'], categorical_feature=CATEGORICAL)
            categories = {c: X_all[c].cat.categories for c in CATEGORICAL}
        elif best_type == 'xgb':
            best_model = make_sklearn_pipeline(XGBRegressor(
                n_estimators=1000, max_depth=7, learning_rate=0.035,
                subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
                reg_alpha=0.1, reg_lambda=3.0, objective='reg:squarederror',
                tree_method='hist', random_state=RANDOM_STATE, n_jobs=-1
            ))
            best_model.fit(data[FEATURES], data['Demand'])
            categories = None
        else:
            best_model = make_sklearn_pipeline(HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                l2_regularization=2.0, random_state=RANDOM_STATE
            ))
            best_model.fit(data[FEATURES], data['Demand'])
            categories = None

        evaluations = [x[2] for x in candidates]
        evaluation_df = pd.DataFrame(evaluations).sort_values(['MAE', 'RMSE']).reset_index(drop=True)
        evaluation_df.to_csv(EVAL_FILE, index=False, encoding='utf-8-sig')
        data.to_csv(PROCESSED_FILE, index=False, encoding='utf-8-sig')
        info = {
            'model': best_name, 'mae': best_eval['MAE'], 'rmse': best_eval['RMSE'],
            'mape': best_eval['MAPE'], 'r2': best_eval['R2'],
            'trained_at': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            'data_hash': file_hash(DATA_FILE), 'rows': len(raw),
            'data_from': str(raw['Date'].min().date()), 'data_to': str(raw['Date'].max().date()),
            'split_date': str(pd.Timestamp(split_date).date()), 'feature_count': len(FEATURES),
            'candidates': evaluations
        }
        bundle = {'model': best_model, 'model_name': best_name, 'categories': categories,
                  'evaluations': evaluations, 'train_test_split_date': str(pd.Timestamp(split_date).date()),
                  'trained_at': info['trained_at'], 'data_hash': info['data_hash'], 'feature_count': len(FEATURES)}
        joblib.dump(bundle, MODEL_FILE)
        MODEL_INFO_FILE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        DATA_INFO_FILE.write_text(json.dumps({'hash': info['data_hash'], 'rows': len(raw), 'updated_at': info['trained_at']}, ensure_ascii=False, indent=2), encoding='utf-8')
        return bundle


def make_future_features(history_demand, history_sold, future_date, price, discount, category, store, product):
    def last(n): return float(history_demand[-n]) if len(history_demand) >= n else float(np.mean(history_demand))
    def mean(n): return float(np.mean(history_demand[-n:]))
    def std(n): return float(np.std(history_demand[-n:])) if len(history_demand[-n:]) > 1 else 0.0
    d7, d28, d90 = mean(7), mean(28), mean(90)
    row = {
        'Store ID': store, 'Product ID': product, 'Category': category, 'Seasonality': season_from_month(future_date.month),
        'Month': future_date.month, 'DayOfWeek': future_date.weekday(), 'WeekOfYear': int(future_date.isocalendar().week),
        'DayOfYear': future_date.dayofyear, 'DayOfMonth': future_date.day, 'Quarter': future_date.quarter,
        'IsWeekend': int(future_date.weekday() >= 5), 'IsMonthStart': int(future_date.is_month_start), 'IsMonthEnd': int(future_date.is_month_end),
        'SinYear': np.sin(2*np.pi*future_date.dayofyear/365.25), 'CosYear': np.cos(2*np.pi*future_date.dayofyear/365.25),
        'SinWeek': np.sin(2*np.pi*future_date.weekday()/7), 'CosWeek': np.cos(2*np.pi*future_date.weekday()/7), 'SinMonth': np.sin(2*np.pi*future_date.month/12),
        **{f'Demand_Lag_{n}': last(n) for n in [1,2,3,7,14,21,28,30,56,90,180,365]},
        **{f'Demand_Mean_{n}': mean(n) for n in [3,7,14,28,56,90,365]},
        'Demand_Median_7': float(np.median(history_demand[-7:])), 'Demand_Median_28': float(np.median(history_demand[-28:])),
        'Demand_Std_7': std(7), 'Demand_Std_28': std(28), 'Demand_Std_56': std(56), 'Demand_Std_90': std(90),
        'Demand_EWM_7': float(pd.Series(history_demand).ewm(span=7, adjust=False).mean().iloc[-1]),
        'Demand_EWM_28': float(pd.Series(history_demand).ewm(span=28, adjust=False).mean().iloc[-1]),
        'Demand_EWM_90': float(pd.Series(history_demand).ewm(span=90, adjust=False).mean().iloc[-1]),
        'Demand_Trend_7_28': d7-d28, 'Demand_Trend_28_90': d28-d90,
        'Sold_Lag_1': float(history_sold[-1]), 'Sold_Lag_7': float(history_sold[-7]) if len(history_sold)>=7 else float(np.mean(history_sold)),
        'Sold_Mean_7': float(np.mean(history_sold[-7:])), 'Price': price, 'Discount': discount
    }
    return pd.DataFrame([row])


def predict_row(bundle, row):
    if bundle['model_name'] == 'LightGBM':
        x = prepare_lgbm(row, bundle.get('categories'))
    else:
        x = row[FEATURES]
    return float(max(0, bundle['model'].predict(x)[0]))


def forecast_inventory(bundle, horizon=7):
    horizon = 30 if int(horizon) == 30 else 7
    raw = load_data(DATA_FILE)
    last_date = raw['Date'].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
    latest = raw.sort_values('Date').groupby(['Store ID','Product ID'], as_index=False).tail(1)
    results = []
    for _, latest_row in latest.iterrows():
        store, product = latest_row['Store ID'], latest_row['Product ID']
        history = raw[(raw['Store ID']==store)&(raw['Product ID']==product)].sort_values('Date')
        demand_history = history['Demand'].astype(float).tolist()
        sold_history = history['Units Sold'].astype(float).tolist()
        inventory = float(latest_row['Inventory Level'])
        preds=[]
        for dt in future_dates:
            row = make_future_features(demand_history, sold_history, dt, float(latest_row['Price']), float(latest_row['Discount']), latest_row['Category'], store, product)
            p = predict_row(bundle, row)
            preds.append(p); demand_history.append(p); sold_history.append(p)
        total = float(sum(preds)); avg = total/horizon
        recent = history['Demand'].tail(30)
        reserve = float(SERVICE_Z * recent.std())
        target = total + reserve
        order = max(0, int(np.ceil(target-inventory)))
        if inventory < total: status='CẦN NHẬP HÀNG'
        elif inventory < target: status='TỒN KHO THẤP'
        elif inventory <= total*2: status='ĐỦ HÀNG'
        else: status='TỒN KHO CAO'
        results.append({'Store ID':store,'Product ID':product,'Category':latest_row['Category'],'Current Inventory':round(inventory,2),
                        f'Forecast {horizon} Days':round(total,2),'Average Daily Demand':round(avg,2),'Safety Stock':round(reserve,2),
                        'Target Stock':round(target,2),'Recommended Order':order,'Status':status})
    out=pd.DataFrame(results).sort_values('Recommended Order',ascending=False).reset_index(drop=True)
    out.to_csv(FORECAST_FILE,index=False,encoding='utf-8-sig')
    return out


def get_bundle():
    return train_model(force=False)


def detect_new_data_and_train():
    last_hash = None
    while True:
        try:
            if DATA_FILE.exists():
                current_hash = file_hash(DATA_FILE)
                if current_hash != last_hash:
                    saved_hash = None
                    if MODEL_INFO_FILE.exists():
                        try: saved_hash = json.loads(MODEL_INFO_FILE.read_text(encoding='utf-8')).get('data_hash')
                        except Exception: pass
                    if current_hash != saved_hash:
                        print('Phát hiện dữ liệu mới -> tự động huấn luyện lại...')
                        train_model(force=True)
                    last_hash = current_hash
        except Exception as e:
            print('Tự động kiểm tra dữ liệu:', e)
        time.sleep(AUTO_CHECK_SECONDS)


app=Flask(__name__)


@app.route('/')
def index():
    horizon=30 if request.args.get('horizon','7')=='30' else 7
    bundle=get_bundle(); forecast=forecast_inventory(bundle,horizon)
    selected_store=request.args.get('store',''); selected_status=request.args.get('status','')
    view=forecast.copy()
    if selected_store: view=view[view['Store ID']==selected_store]
    if selected_status: view=view[view['Status']==selected_status]
    stores=sorted(forecast['Store ID'].unique().tolist()); forecast_col=f'Forecast {horizon} Days'
    info=json.loads(MODEL_INFO_FILE.read_text(encoding='utf-8')) if MODEL_INFO_FILE.exists() else {}
    summary={'total_products':len(forecast),'need_order':int((forecast['Recommended Order']>0).sum()),'total_order':int(forecast['Recommended Order'].sum()),
             'total_inventory':round(float(forecast['Current Inventory'].sum()),2),'total_forecast':round(float(forecast[forecast_col].sum()),2)}
    return render_template('index.html',rows=view.to_dict('records'),stores=stores,selected_store=selected_store,selected_status=selected_status,
                           summary=summary,model_name=bundle['model_name'],evaluations=bundle['evaluations'],horizon=horizon,forecast_col=forecast_col,info=info)


@app.route('/api/forecast')
def api_forecast():
    horizon=30 if request.args.get('horizon','7')=='30' else 7
    forecast=forecast_inventory(get_bundle(),horizon)
    store=request.args.get('store',''); status=request.args.get('status','')
    if store: forecast=forecast[forecast['Store ID']==store]
    if status: forecast=forecast[forecast['Status']==status]
    return jsonify(forecast.to_dict('records'))


@app.route('/train',methods=['POST'])
def retrain():
    bundle=train_model(force=True); forecast=forecast_inventory(bundle,7)
    return jsonify({'success':True,'model':bundle['model_name'],'evaluations':bundle['evaluations'],'rows':len(forecast)})


@app.route('/upload-data',methods=['POST'])
def upload_data():
    file=request.files.get('file')
    mode=request.form.get('mode','append')
    if not file or not file.filename.lower().endswith('.csv'):
        return jsonify({'success':False,'message':'Vui lòng chọn một file CSV.'}),400
    temp=UPLOAD_DIR/'du_lieu_moi_tam.csv'
    file.save(temp)
    try:
        new_df=load_data(temp)
        if mode == 'append' and DATA_FILE.exists():
            old_df=load_data(DATA_FILE)
            merged=pd.concat([old_df,new_df],ignore_index=True).drop_duplicates().sort_values(['Store ID','Product ID','Date']).reset_index(drop=True)
            backup=UPLOAD_DIR/f"du_lieu_truoc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            DATA_FILE.replace(backup)
            merged.to_csv(DATA_FILE,index=False,encoding='utf-8-sig')
            message=f'Đã nối thêm dữ liệu: {len(new_df):,} dòng mới; tổng dữ liệu hiện có {len(merged):,} dòng.'
            total_rows=len(merged)
        else:
            backup=UPLOAD_DIR/f"du_lieu_truoc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            if DATA_FILE.exists(): DATA_FILE.replace(backup)
            temp.replace(DATA_FILE)
            message=f'Đã thay toàn bộ dữ liệu bằng {len(new_df):,} dòng.'
            total_rows=len(new_df)
        if temp.exists(): temp.unlink()
        bundle=train_model(force=True)
        forecast_inventory(bundle,7)
        return jsonify({'success':True,'message':message+' Hệ thống đã tự động huấn luyện lại.','model':bundle['model_name'],'rows':total_rows,'data_to':str(load_data(DATA_FILE)['Date'].max().date())})
    except Exception as e:
        if temp.exists(): temp.unlink()
        return jsonify({'success':False,'message':str(e)}),400


@app.route('/data-status')
def data_status():
    info=json.loads(MODEL_INFO_FILE.read_text(encoding='utf-8')) if MODEL_INFO_FILE.exists() else {}
    return jsonify(info)


@app.route('/download')
def download_csv():
    horizon=30 if request.args.get('horizon','7')=='30' else 7
    forecast_inventory(get_bundle(),horizon)
    return send_file(FORECAST_FILE,as_attachment=True,download_name=f'du_bao_nhap_kho_{horizon}_ngay.csv')


if __name__=='__main__':
    print('='*65); print('HỆ THỐNG DỰ BÁO NHU CẦU VÀ GỢI Ý NHẬP KHO'); print('='*65)
    bundle=get_bundle(); print('Mô hình được chọn:',bundle['model_name'])
    t=threading.Thread(target=detect_new_data_and_train,daemon=True); t.start()
    print('Tự kiểm tra dữ liệu mới mỗi',AUTO_CHECK_SECONDS,'giây')
    print('Mở: http://127.0.0.1:5000')
    app.run(host='127.0.0.1',port=5000,debug=False)
