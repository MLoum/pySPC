
import multiprocessing as mp
import time

def do_something(arg, arg2):
    print ("Running thread! Args:", (arg, arg2))
    time.sleep(3)
    print ("Done!")

if __name__ == '__main__':

    nb_of_workers = 6
    nb_of_chunk = 15

    arg = []
    arg2 = []

    for i in range(nb_of_chunk):
        arg.append(i)
        arg2.append(i*i)


    p = mp.Pool(processes=nb_of_workers)

    method = "async"
    if method == "apply":
        results = [p.apply(do_something, args=(arg[i], arg2[i])) for i
                  in range(nb_of_chunk)]
    elif method == "async":
        results = [p.apply_async(do_something, args=(arg[i], arg2[i])) for i
                  in range(nb_of_chunk)]
        output = [p.get() for p in results]
        print(output)



