import multiprocessing
from multiprocessing import Pool

def worker(num):
    """thread worker function"""
    print ('Worker:', num)
    return





def fct(x):
    return x ** 2

if __name__ == '__main__':
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i,))
        jobs.append(p)
        p.start()


    pool = Pool(4)
    for res in pool.map(fct, range(20)):
        print(res)


