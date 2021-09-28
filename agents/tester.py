import time
import importlib
import numpy as np

import torch
import torch.nn.functional as F

from agents.base import NNAgent
from utils.metrics import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf

from tqdm import tqdm


class Tester(NNAgent):
    def __init__(self, config):
        super().__init__(config)

        if(self.config.test.get('preprocessing_function', '') != ""):
            self.config.test.preprocessing_function = importlib.import_module(
                'datasets.preprocessing.functions').__getattribute__(self.config.test.preprocessing_function)
        else:
            self.config.test.preprocessing_function = None
        
        TestDataset = importlib.import_module(
            'datasets.' + self.config.test.dataset).__getattribute__(self.config.test.dataset)
        self.test_dataset = TestDataset(**vars(self.config.test))
        self.loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.config.test.nDataLoaderThread,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            # sampler=self.test.sampler
        )
        self.test_normalize = self.config.test.get('normalize', True)


    def run(self):
        try:
            if self.print_metrics:
                self.get_metrics()
            self.validate()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C... Wait to finalize.")

    def validate(self):
        self.__model__.eval()

        all_scores = []
        all_labels = []

        eer = 0
        mindcf = 0

        total_iterations = len(self.loader)
        current_iteration = 0
        start_time = time.time()

        tqdm_test = tqdm(self.loader, total=len(
            self.loader), desc="Epoch {}".format(self.current_epoch+1))
        for ref, com, label in tqdm_test:
            current_iteration += 1

            loop_time = time.time()-start_time

            label = int(label.data.cpu().numpy()[0])
            with torch.no_grad():
                # Feature extraction
                ref_feat = self.__model__.get_feat(ref, self.test_normalize)
                com_feat = self.__model__.get_feat(com, self.test_normalize)

                # Scoring by distance
                score = F.pairwise_distance(ref_feat, com_feat)
                score = score.detach().cpu().numpy()
                score = -1 * np.mean(score)

            all_scores.append(score)
            all_labels.append(label)

            if (current_iteration % self.config.test.print_interval == 0) or (current_iteration == total_iterations):
                tunedThreshold, eer, fpr, fnr = tuneThresholdfromScore(
                    all_scores, all_labels, [1, 0.1])
                p_target = 0.05
                c_miss = 1
                c_fa = 1
                fnrs, fprs, thresholds = ComputeErrorRates(
                    all_scores, all_labels)
                mindcf, threshold = ComputeMinDcf(
                    fnrs, fprs, thresholds, p_target, c_miss, c_fa)

            process_time = time.time()-start_time-loop_time

            # Logging
            total = process_time+loop_time
            perc_proc = process_time/total*100

            tqdm_test.set_description("Epoch-{} Validation | VEER {:2.4f}% MDC {:2.5f} | Efficiency {:2.2f}% "
                                      .format(self.current_epoch+1, eer, mindcf, perc_proc))

            start_time = time.time()

        validation_time = tqdm_test.format_dict['elapsed']
        validation_rate = total_iterations / validation_time

        tqdm_test.close()

        self.logger.info("Validation at epoch-{} completed in {:2.1f}s ({:2.1f}samples/s). VEER {:2.4f}% MDC {:2.5f} ".format(
            self.current_epoch+1, validation_time, validation_rate, eer, mindcf))

        return eer
