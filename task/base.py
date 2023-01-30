from typing import List, Dict


class BaseTask:


    def get_train_dataset(self):
        pass

    def get_eval_dataset(self, x):
        pass

    def get_test_dataset(self, x):
        pass

    def training_step(self, batch):
        """
            return loss
        """
        pass

    def evaluation_step(self, batch):
        """
            return dict
        """
        pass

    def collate_evaluation(self, results: List[Dict]):
        """
            return dict(metric)
        """
        pass