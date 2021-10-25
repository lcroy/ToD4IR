import os

class Config:
    def __init__(self):
        # project path
        self.project_path = os.path.dirname(os.path.abspath(__file__))

        # models path
        self.model_path = os.path.join(self.project_path, 'train_models')
        self.model_gpt_neo_checkpoint_path = os.path.join(self.model_path, 'gpt_neo_checkpoint')
        self.model_gpt_neo_tokenizer_path = os.path.join(self.model_path, 'gpt_neo_checkpoint')
        self.model_gpt_neo_log_path = os.path.join(self.project_path, 'logs')
        self.model_gpt2_checkpoint_path = os.path.join(self.model_path, 'gpt2_checkpoint')

        # data
        # for model training
        self.dataset_path = os.path.join(self.project_path, 'dataset')
        self.dataset_path_IR = os.path.join(self.dataset_path, 'Industrial-robots/data_xiaochun.json')
        self.dataset_path_IR_delex = os.path.join(self.dataset_path, 'Industrial-robots/train.context.belief.sysact.tres.sres.delex')
        self.dataset_path_IR_pre_delex = os.path.join(self.dataset_path,
                                                'Industrial-robots/train.context.belief.sysact.tres.sres.pre.delex')
        self.dataset_path_IR_val_delex = os.path.join(self.dataset_path,
                                                  'Industrial-robots/train.context.belief.sysact.tres.sres.delex')
        self.dataset_path_production_db = os.path.join(self.dataset_path, 'db/IRDB.db')
        # for testing
        self.dataset_path_decoded_file = os.path.join(self.dataset_path, 'Industrial-robots/decoded_file')
        self.dataset_path_test_file = os.path.join(self.dataset_path, 'Industrial-robots/test_file')
        # generated from IRWOZ interface
        self.dataset_path_gen_IRWOZ = os.path.join(self.dataset_path, 'Industrial-robots/gen_dialogue.json')

        # parameters
        self.max_length = 1024

        # response
        self.stop_words = ['bye', 'goodbye', 'stop', 'exit']
        self.max_end_dialogue = ['sure, goodbye.', 'Bye.', 'talk to you soon.']
