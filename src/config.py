import os


class Config:
    module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(module_path, 'data')
    training_data_path = os.path.join(data_path, 'training')
    test_data_path = os.path.join(data_path, 'test')
