import random
import multiprocessing

import numpy as np
from tqdm import tqdm
import time


def generate_unique_combinations(total_elements, combination_size):
    if combination_size > total_elements:
        raise ValueError("Combination size cannot be greater than the total number of elements")

    used_combinations = set()

    def generate_combination():
        nonlocal used_combinations

        n_try = 0
        current_combination = None
        while n_try < 5000:
            current_combination = sorted(random.sample(range(total_elements), combination_size))
            current_tuple = tuple(current_combination)

            if current_tuple not in used_combinations:
                used_combinations.add(current_tuple)
                return current_combination

            n_try += 1
        return current_combination

    return generate_combination


class ProcessPool:
    def __init__(self, num_processes, verbose=True):
        self.num_processes = num_processes
        self.verbose = verbose
        if self.num_processes is None or self.num_processes <= 0:
            self.num_processes = multiprocessing.cpu_count()
        self.pool = multiprocessing.Pool(num_processes)
        self.results = []

    def apply_async(self, func, args=(), kwargs=None, callback=None):
        if kwargs is None:
            kwargs = {}
        result = self.pool.apply_async(func, args=args, kwds=kwargs, callback=callback)
        self.results.append(result)

    def progress_bar(self, desc=None):
        results = []
        if self.verbose:
            with tqdm(total=len(self.results), desc=desc) as pbar:
                while self.results:
                    completed = []
                    for result in self.results:
                        if result.ready():
                            results.append(result.get())
                            completed.append(result)
                        else:
                            break
                    for result in completed:
                        self.results.remove(result)
                    pbar.update(len(completed))
        else:
            while self.results:
                completed = []
                for result in self.results:
                    if result.ready():
                        results.append(result.get())
                        completed.append(result)
                    else:
                        break
                for result in completed:
                    self.results.remove(result)
        return results

    def close(self):
        self.pool.close()
        self.pool.join()


def task(x):
    # 执行任务
    time.sleep(x)
    return x * x


if __name__ == '__main__':
    pass



