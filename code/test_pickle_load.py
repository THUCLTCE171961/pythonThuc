import joblib
# hoặc
import pickle

for file in ['motion_svm_model.pkl', 'motion_scaler.pkl', 'static_dynamic_classifier.pkl']:
    path = f'user_69181d5716148ced5c4367e5/models/{file}'
    try:
        try:
            obj = joblib.load(path)
            print(f'{file}: Đọc joblib thành công!')
        except Exception as e_joblib:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            print(f'{file}: Đọc pickle thành công!')
    except Exception as e:
        print(f'{file}: lỗi {e}')