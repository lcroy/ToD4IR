import os


class Config:
    def __init__(self):
        # project path
        self.project_path = os.path.dirname(os.path.abspath(__file__))

        # models path
        self.model_path = os.path.join(self.project_path, 'train_models')
        self.model_checkpoint_path = os.path.join(self.model_path, 'checkpoint')
        # self.main_model_LSTM_path = os.path.join(self.model_path, 'main_intent/main.pt')
        # self.main_model_BERT_path = os.path.join(self.model_path, 'main_intent/main_intent.pt')
        # self.mir_model_path = os.path.join(self.model_path, 'mir')

        # data
        # for model training
        self.dataset_path = os.path.join(self.project_path, 'dataset')
        self.dataset_path_IR = os.path.join(self.dataset_path, 'Industrial-robots/data_chen.json')
        self.dataset_path_IR_lex = os.path.join(self.dataset_path, 'Industrial-robots/train.context.belief.sysact.tres.sres.lex')
        self.dataset_path_IR_delex = os.path.join(self.dataset_path,
                                                'Industrial-robots/train.context.belief.sysact.tres.sres.delex')
        self.dataset_path_IR_val_delex = os.path.join(self.dataset_path,
                                                  'Industrial-robots/train.context.belief.sysact.tres.sres.delex')

        # for real time prediction
        self.dataset_path_production_db = 'C:/Users/lcroy/OneDrive/Desktop/IRDB.db'


        # parameters
        self.max_length = 1024

        #
        # self.main_intent_dataset_path = os.path.join(self.dataset_path, 'main_intent')
        # self.main_intent_dataset_csv = os.path.join(self.main_intent_dataset_path, 'train/main_intent.tsv')
        # self.main_intent_label_csv = os.path.join(self.main_intent_dataset_path, 'main_intent_label.tsv')
        # self.hint_sound = os.path.join(self.dataset_path, 'SR/Balloon.mp3')
        # self.SR_json = os.path.join(self.dataset_path, 'SR/polybottest-firebase-key.json')
        # self.vocab_json = os.path.join(self.main_intent_dataset_path, 'vocab.json')
