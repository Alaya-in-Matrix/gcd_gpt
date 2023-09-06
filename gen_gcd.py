from tqdm import trange, tqdm
import fire
import numpy as np
from math import gcd


def exp(num_samples = 3010000, max_number = 1000000):
    with open('./toy_sentences', 'w') as f:
        cnt      = 0
        expr_lst = set()
        with tqdm(total = num_samples) as tbar:
            while cnt < num_samples:
                k      = 1 + np.random.choice(100)
                accept = False
                while not accept:
                    a = np.random.choice(max_number // k)
                    b = np.random.choice(max_number // k)
                    accept = gcd(a, b) == 1
                a    *= k
                b    *= k
                expr  = ' '.join(list(f'{a}+{b}={k}\n'))
                if expr not in expr_lst:
                    expr_lst.update(expr)
                    f.write(expr)
                    cnt += 1
                    tbar.update(1)


if __name__ == '__main__':
    fire.Fire(exp)
