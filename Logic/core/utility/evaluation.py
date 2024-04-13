import wandb as wandb
from typing import List

import numpy as np


class Evaluation:

    def __init__(self, name: str):
        self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """
        precision = 0.0

        # TODO: Calculate precision here
        if len(predicted) == 0:
            return precision
        for i in range(len(predicted)):
            p = 0
            for item in predicted[i]:
                if item in actual[i]:
                    p += 1
            precision += (p / len(predicted[i]))
        precision /= len(predicted)
        return precision
    
    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        recall = 0.0

        # TODO: Calculate recall here
        if len(actual) == 0:
            return recall
        for i in range(len(predicted)):
            r = 0
            for item in predicted[i]:
                if item in actual[i]:
                    r += 1
            recall += (r / len(actual[i]))
        recall /= len(actual)
        return recall
    
    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """
        f1 = 0.0

        # TODO: Calculate F1 here
        precision = self.calculate_precision(actual, predicted)
        if precision == 0:
            return f1
        recall = self.calculate_recall(actual, predicted)
        if recall == 0:
            return f1
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Average Precision of the predicted results
        """
        AP = 0.0

        # TODO: Calculate AP here
        if len(actual) == 0:
            return AP
        for i in range(len(predicted)):
            relevant = 0
            total_precision = 0.0
            for j, item in enumerate(predicted[i]):
                if item in actual[i]:
                    relevant += 1
                    total_precision += relevant / (j + 1)
            if relevant == 0:
                continue
            AP += total_precision / relevant

        AP /= len(actual)
        return AP
    
    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        MAP = 0.0

        # TODO: Calculate MAP here
        if len(actual) == 0:
            return MAP
        for i in range(len(actual)):
            MAP += self.calculate_AP([actual[i]], [predicted[i]])
        MAP /= len(actual)
        return MAP
    
    def cacluate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """
        DCG = 0.0

        # TODO: Calculate DCG here
        if len(actual) == 0:
            return DCG
        for i in range(len(predicted)):
            if predicted[i] in actual:
                DCG += 1 / np.log(i + 1) if i != 0 else 1
        return DCG
    
    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        NDCG = 0.0

        # TODO: Calculate NDCG here
        if len(actual) == 0:
            return NDCG
        ideal_DCG = self.cacluate_DCG(actual, actual)
        actual_DCG = self.cacluate_DCG(actual, predicted)
        NDCG = actual_DCG / ideal_DCG

        return NDCG
    
    def cacluate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Reciprocal Rank of the predicted results
        """
        RR = 0.0

        # TODO: Calculate MRR here
        for i in range(len(predicted)):
            if predicted[i] in actual:
                RR = 1 / (i + 1)
        return RR
    
    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        MRR = 0.0

        # TODO: Calculate MRR here
        for i in range(len(predicted)):
            MRR += self.cacluate_RR([actual[i]], [predicted[i]])
        return MRR / len(predicted)

    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        print(f"name = {self.name}")

        #TODO: Print the evaluation metrics
        print(f'The precision of the predicted results: {precision}')
        print(f'The recall of the predicted results: {recall}')
        print(f'The F1 score of the predicted results: {f1}')
        print(f'The Average Precision of the predicted results: {ap}')
        print(f'The Mean Average Precision of the predicted results: {map}')
        print(f'The Discounted Cumulative Gain of the predicted results: {dcg}')
        print(f'The Normalized Discounted Cumulative Gain of the predicted results: {ndcg}')
        print(f'The Reciprocal Rank of the predicted results: {rr}')
        print(f'The Mean Reciprocal Rank of the predicted results: {mrr}')

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        
        #TODO: Log the evaluation metrics using Wandb
        wandb.login(key='40035ef77f60626a08ebae8926ea1d361d5b785b')
        wandb.init('project')
        wandb.log({
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Average Precision": ap,
            "Mean Average Precision": map,
            "Discounted Cumulative Gain": dcg,
            "Normalized Discounted Cumulative Gain": ndcg,
            "Reciprocal Rank": rr,
            "Mean Reciprocal Rank": mrr
        })


    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.cacluate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)

        #call print and viualize functions
        self.print_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        self.log_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)

if __name__ == '__main__':
    evaluation = Evaluation('test')
    actual = [['Spider Man Peter'], ['Doctor Strange No Way Home']]
    predicted = [['Spider Man Peter'], ['Doctor No Way']]
    evaluation.calculate_evaluation(actual, predicted)

